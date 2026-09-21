use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use numpy::ndarray::{Array1, Array2, Array3, ShapeBuilder};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList};

#[derive(Clone, Debug)]
struct Rle {
    h: usize,
    w: usize,
    counts: Vec<u32>,
}

fn value_error(msg: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.into())
}

fn counts_to_bytes(counts: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(counts.len() * 2);
    for i in 0..counts.len() {
        let mut x = counts[i] as i64;
        if i > 2 {
            x -= counts[i - 2] as i64;
        }
        loop {
            let mut c = (x & 0x1f) as u8;
            x >>= 5;
            let more = if c & 0x10 != 0 { x != -1 } else { x != 0 };
            if more {
                c |= 0x20;
            }
            c += 48;
            out.push(c);
            if !more {
                break;
            }
        }
    }
    out
}

fn bytes_to_counts(bytes: &[u8]) -> PyResult<Vec<u32>> {
    let mut counts: Vec<u32> = Vec::new();
    let mut p = 0usize;
    while p < bytes.len() && bytes[p] != 0 {
        let mut x: i64 = 0;
        let mut k = 0usize;
        loop {
            if p >= bytes.len() {
                return Err(value_error("truncated compressed RLE counts"));
            }
            let c = (bytes[p] as i64) - 48;
            if !(0..=63).contains(&c) {
                return Err(value_error("invalid compressed RLE character"));
            }
            x |= (c & 0x1f) << (5 * k);
            let more = c & 0x20;
            p += 1;
            k += 1;
            if more == 0 {
                if c & 0x10 != 0 {
                    x |= -1i64 << (5 * k);
                }
                break;
            }
        }
        if counts.len() > 2 {
            x += counts[counts.len() - 2] as i64;
        }
        if x < 0 || x > u32::MAX as i64 {
            return Err(value_error("compressed RLE decoded outside uint32 range"));
        }
        counts.push(x as u32);
    }
    Ok(counts)
}

fn rle_to_object(py: Python<'_>, rle: &Rle) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("size", vec![rle.h, rle.w])?;
    d.set_item("counts", PyBytes::new_bound(py, &counts_to_bytes(&rle.counts)))?;
    Ok(d.into_py(py))
}

fn extract_count_bytes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(b) = obj.downcast::<PyBytes>() {
        return Ok(b.as_bytes().to_vec());
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(s.into_bytes());
    }
    Err(value_error("compressed RLE counts must be bytes or str"))
}

fn rle_from_object(obj: &Bound<'_, PyAny>) -> PyResult<Rle> {
    let d = obj.downcast::<PyDict>()?;
    let size_obj = d.get_item("size")?.ok_or_else(|| value_error("RLE missing size"))?;
    let counts_obj = d.get_item("counts")?.ok_or_else(|| value_error("RLE missing counts"))?;
    let size: Vec<usize> = size_obj.extract()?;
    if size.len() != 2 {
        return Err(value_error("RLE size must contain [height, width]"));
    }
    let bytes = extract_count_bytes(&counts_obj)?;
    let counts = bytes_to_counts(&bytes)?;
    Ok(Rle { h: size[0], w: size[1], counts })
}

fn rles_from_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Rle>> {
    let list = obj.downcast::<PyList>()?;
    list.iter().map(|item| rle_from_object(&item)).collect()
}

fn area_one(rle: &Rle) -> u32 {
    rle.counts.iter().skip(1).step_by(2).copied().sum()
}

fn bbox_one(rle: &Rle) -> [f64; 4] {
    let h = rle.h as u32;
    let w = rle.w as u32;
    let m = (rle.counts.len() / 2) * 2;
    if m == 0 || h == 0 || w == 0 {
        return [0.0; 4];
    }
    let mut xs = w;
    let mut ys = h;
    let mut xe = 0u32;
    let mut ye = 0u32;
    let mut xp = 0u32;
    let mut cc = 0u32;
    for j in 0..m {
        cc = cc.saturating_add(rle.counts[j]);
        let t = cc.saturating_sub((j % 2) as u32);
        let y = t % h;
        let x = (t - y) / h;
        if j % 2 == 0 {
            xp = x;
        } else if xp < x {
            ys = 0;
            ye = h - 1;
        }
        xs = xs.min(x);
        xe = xe.max(x);
        ys = ys.min(y);
        ye = ye.max(y);
    }
    [xs as f64, ys as f64, (xe - xs + 1) as f64, (ye - ys + 1) as f64]
}

fn bbox_iou_xywh(dt: &[f64; 4], gt: &[f64; 4], crowd: bool) -> f64 {
    let da = dt[2] * dt[3];
    let ga = gt[2] * gt[3];
    let w = (dt[2] + dt[0]).min(gt[2] + gt[0]) - dt[0].max(gt[0]);
    if w <= 0.0 {
        return 0.0;
    }
    let h = (dt[3] + dt[1]).min(gt[3] + gt[1]) - dt[1].max(gt[1]);
    if h <= 0.0 {
        return 0.0;
    }
    let inter = w * h;
    let union = if crowd { da } else { da + ga - inter };
    if union == 0.0 { 0.0 } else { inter / union }
}

fn rle_iou_runs(dt: &Rle, gt: &Rle, crowd: bool, dt_area: u32) -> f64 {
    // Exact RLE overlap for a pair that has already passed the inexpensive
    // bounding-box overlap prefilter.  Keep this inner loop free of geometry
    // setup: the matrix-level caller precomputes bbox / area values once per
    // instance, matching the structure of the original COCO C kernel.
    if dt.h != gt.h || dt.w != gt.w {
        return -1.0;
    }
    let mut a = 1usize;
    let mut b = 1usize;
    let mut ca = *dt.counts.get(0).unwrap_or(&0);
    let mut cb = *gt.counts.get(0).unwrap_or(&0);
    let mut va = false;
    let mut vb = false;
    let mut inter = 0u32;
    let mut union = 0u32;
    loop {
        let c = ca.min(cb);
        if va || vb {
            union = union.saturating_add(c);
            if va && vb {
                inter = inter.saturating_add(c);
            }
        }
        ca -= c;
        cb -= c;
        if ca == 0 && a < dt.counts.len() {
            ca = dt.counts[a];
            a += 1;
            va = !va;
        }
        if cb == 0 && b < gt.counts.len() {
            cb = gt.counts[b];
            b += 1;
            vb = !vb;
        }
        if ca == 0 && cb == 0 && a >= dt.counts.len() && b >= gt.counts.len() {
            break;
        }
    }
    if inter == 0 {
        union = 1;
    } else if crowd {
        union = dt_area;
    }
    inter as f64 / union as f64
}

fn encode_plane<F>(h: usize, w: usize, mut pixel: F) -> Rle
where
    F: FnMut(usize, usize) -> u8,
{
    let mut counts = Vec::with_capacity(h * w / 4 + 2);
    let mut previous = 0u8;
    let mut count = 0u32;
    for x in 0..w {
        for y in 0..h {
            let value = if pixel(y, x) == 0 { 0 } else { 1 };
            if value != previous {
                counts.push(count);
                count = 0;
                previous = value;
            }
            count += 1;
        }
    }
    counts.push(count);
    Rle { h, w, counts }
}

fn encode_plane_memory_order(h: usize, w: usize, data: &[u8]) -> Rle {
    debug_assert_eq!(data.len(), h * w);
    let mut counts = Vec::with_capacity(h * w / 4 + 2);
    let mut previous = 0u8;
    let mut count = 0u32;
    for &pixel in data {
        let value = if pixel == 0 { 0 } else { 1 };
        if value != previous {
            counts.push(count);
            count = 0;
            previous = value;
        }
        count += 1;
    }
    counts.push(count);
    Rle { h, w, counts }
}

fn merge_pair(a: &Rle, b: &Rle, intersect: bool) -> Rle {
    if a.h != b.h || a.w != b.w {
        return Rle { h: 0, w: 0, counts: Vec::new() };
    }
    let mut ia = 1usize;
    let mut ib = 1usize;
    let mut ca = *a.counts.get(0).unwrap_or(&0);
    let mut cb = *b.counts.get(0).unwrap_or(&0);
    let mut va = false;
    let mut vb = false;
    let mut out_value = false;
    let mut run = 0u32;
    let mut counts = Vec::new();
    loop {
        let c = ca.min(cb);
        run = run.saturating_add(c);
        ca -= c;
        cb -= c;
        if ca == 0 && ia < a.counts.len() {
            ca = a.counts[ia];
            ia += 1;
            va = !va;
        }
        if cb == 0 && ib < b.counts.len() {
            cb = b.counts[ib];
            ib += 1;
            vb = !vb;
        }
        let next_value = if intersect { va && vb } else { va || vb };
        let done = ca == 0 && cb == 0 && ia >= a.counts.len() && ib >= b.counts.len();
        if next_value != out_value || done {
            counts.push(run);
            run = 0;
            out_value = next_value;
        }
        if done {
            break;
        }
    }
    Rle { h: a.h, w: a.w, counts }
}

fn polygon_to_rle(poly: &[f64], h: usize, w: usize) -> Rle {
    let k = poly.len() / 2;
    if k < 3 || h == 0 || w == 0 {
        return Rle { h, w, counts: vec![(h * w) as u32] };
    }
    let scale = 5.0f64;
    let mut x = vec![0i32; k + 1];
    let mut y = vec![0i32; k + 1];
    for j in 0..k {
        x[j] = (scale * poly[j * 2] + 0.5) as i32;
        y[j] = (scale * poly[j * 2 + 1] + 0.5) as i32;
    }
    x[k] = x[0];
    y[k] = y[0];
    let mut u: Vec<i32> = Vec::new();
    let mut v: Vec<i32> = Vec::new();
    for j in 0..k {
        let mut xs = x[j];
        let mut xe = x[j + 1];
        let mut ys = y[j];
        let mut ye = y[j + 1];
        let dx = (xe - xs).abs();
        let dy = (ys - ye).abs();
        let flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
        if flip {
            std::mem::swap(&mut xs, &mut xe);
            std::mem::swap(&mut ys, &mut ye);
        }
        if dx >= dy {
            if dx == 0 {
                u.push(xs);
                v.push(ys);
            } else {
                let s = (ye - ys) as f64 / dx as f64;
                for d in 0..=dx {
                    let tmp = if flip { dx - d } else { d };
                    u.push(tmp + xs);
                    v.push((ys as f64 + s * tmp as f64 + 0.5) as i32);
                }
            }
        } else {
            let s = (xe - xs) as f64 / dy as f64;
            for d in 0..=dy {
                let tmp = if flip { dy - d } else { d };
                v.push(tmp + ys);
                u.push((xs as f64 + s * tmp as f64 + 0.5) as i32);
            }
        }
    }
    let mut boundary: Vec<u32> = Vec::new();
    for j in 1..u.len() {
        if u[j] != u[j - 1] {
            let mut xd = if u[j] < u[j - 1] { u[j] as f64 } else { (u[j] - 1) as f64 };
            xd = (xd + 0.5) / scale - 0.5;
            if xd.floor() != xd || xd < 0.0 || xd > (w as f64 - 1.0) {
                continue;
            }
            let mut yd = if v[j] < v[j - 1] { v[j] as f64 } else { v[j - 1] as f64 };
            yd = (yd + 0.5) / scale - 0.5;
            yd = yd.max(0.0).min(h as f64).ceil();
            boundary.push((xd as u32) * (h as u32) + yd as u32);
        }
    }
    boundary.push((h * w) as u32);
    boundary.sort_unstable();
    let mut deltas = Vec::with_capacity(boundary.len());
    let mut p = 0u32;
    for t in boundary {
        deltas.push(t - p);
        p = t;
    }
    let mut counts = Vec::new();
    let mut j = 0usize;
    if !deltas.is_empty() {
        counts.push(deltas[0]);
        j = 1;
    }
    while j < deltas.len() {
        if deltas[j] > 0 {
            counts.push(deltas[j]);
            j += 1;
        } else {
            j += 1;
            if j < deltas.len() {
                if let Some(last) = counts.last_mut() {
                    *last += deltas[j];
                }
                j += 1;
            }
        }
    }
    Rle { h, w, counts }
}

#[pyfunction]
pub fn encode<'py>(
    py: Python<'py>,
    mask: PyReadonlyArray3<'py, u8>,
) -> PyResult<Vec<PyObject>> {
    let arr = mask.as_array();
    let shape = arr.shape();
    let (h, w, n) = (shape[0], shape[1], shape[2]);
    let mut result = Vec::with_capacity(n);

    // COCO masks normally arrive in Fortran order: y is the fastest-moving
    // coordinate, then x, then the mask index. Reversing the axes turns that
    // same allocation into a standard-layout (n, w, h) view, which lets us
    // scan each plane as one contiguous byte slice without per-pixel ndarray
    // indexing. Arbitrary strided inputs retain the general fallback below.
    let reversed = arr.view().reversed_axes();
    if reversed.is_standard_layout() {
        let data = reversed
            .as_slice()
            .expect("standard-layout reversed mask must expose a slice");
        let plane_size = h * w;
        for i in 0..n {
            let start = i * plane_size;
            let stop = start + plane_size;
            let rle = encode_plane_memory_order(h, w, &data[start..stop]);
            result.push(rle_to_object(py, &rle)?);
        }
    } else {
        for i in 0..n {
            let rle = encode_plane(h, w, |y, x| arr[[y, x, i]]);
            result.push(rle_to_object(py, &rle)?);
        }
    }
    Ok(result)
}

#[pyfunction]
pub fn decode<'py>(
    py: Python<'py>,
    rle_objs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let rles = rles_from_list(rle_objs)?;
    if rles.is_empty() {
        return Ok(Array3::<u8>::zeros((0, 0, 0)).into_pyarray_bound(py));
    }
    let h = rles[0].h;
    let w = rles[0].w;
    if rles.iter().any(|r| r.h != h || r.w != w) {
        return Err(value_error("all RLEs must have the same dimensions for decode"));
    }
    let n = rles.len();
    let mut data = vec![0u8; h * w * n];
    for (i, rle) in rles.iter().enumerate() {
        let mut offset = 0usize;
        let mut value = 0u8;
        for &run in &rle.counts {
            for _ in 0..run as usize {
                if offset >= h * w {
                    return Err(value_error("RLE run lengths exceed image dimensions"));
                }
                data[i * h * w + offset] = value;
                offset += 1;
            }
            value ^= 1;
        }
        if offset != h * w {
            return Err(value_error("RLE run lengths do not fill image dimensions"));
        }
    }
    let arr = Array3::from_shape_vec((h, w, n).f(), data)
        .map_err(|e| value_error(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (rle_objs, intersect=0))]
pub fn merge(
    py: Python<'_>,
    rle_objs: &Bound<'_, PyAny>,
    intersect: i32,
) -> PyResult<PyObject> {
    let rles = rles_from_list(rle_objs)?;
    let out = if rles.is_empty() {
        Rle { h: 0, w: 0, counts: Vec::new() }
    } else {
        let mut acc = rles[0].clone();
        for other in rles.iter().skip(1) {
            acc = merge_pair(&acc, other, intersect != 0);
        }
        acc
    };
    rle_to_object(py, &out)
}

#[pyfunction]
pub fn area<'py>(
    py: Python<'py>,
    rle_objs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let rles = rles_from_list(rle_objs)?;
    let values: Vec<u32> = rles.iter().map(area_one).collect();
    Ok(Array1::from_vec(values).into_pyarray_bound(py))
}

fn extract_boxes(obj: &Bound<'_, PyAny>) -> PyResult<Option<Vec<[f64; 4]>>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f64>>() {
        let view = arr.as_array();
        if view.ncols() != 4 {
            return Err(value_error(format!("bbox ndarray must have shape Nx4, got {:?}", view.shape())));
        }
        return Ok(Some((0..view.nrows()).map(|i| {
            let r = view.row(i);
            [r[0], r[1], r[2], r[3]]
        }).collect()));
    }
    Ok(None)
}

#[pyfunction]
pub fn iou<'py>(
    py: Python<'py>,
    dt: &Bound<'py, PyAny>,
    gt: &Bound<'py, PyAny>,
    iscrowd: Vec<u8>,
) -> PyResult<PyObject> {
    if let (Some(dt_boxes), Some(gt_boxes)) = (extract_boxes(dt)?, extract_boxes(gt)?) {
        if iscrowd.len() != gt_boxes.len() {
            return Err(value_error("iscrowd length must equal number of ground truths"));
        }
        if dt_boxes.is_empty() || gt_boxes.is_empty() {
            return Ok(Vec::<f64>::new().into_py(py));
        }
        let mut out = Array2::<f64>::zeros((dt_boxes.len(), gt_boxes.len()));
        for d in 0..dt_boxes.len() {
            for g in 0..gt_boxes.len() {
                out[(d, g)] = bbox_iou_xywh(&dt_boxes[d], &gt_boxes[g], iscrowd[g] != 0);
            }
        }
        return Ok(out.into_pyarray_bound(py).into_py(py));
    }

    let dt_rles = rles_from_list(dt)?;
    let gt_rles = rles_from_list(gt)?;
    if iscrowd.len() != gt_rles.len() {
        return Err(value_error("iscrowd length must equal number of ground truths"));
    }
    if dt_rles.is_empty() || gt_rles.is_empty() {
        return Ok(Vec::<f64>::new().into_py(py));
    }

    // The original COCO C implementation computes each RLE's bbox once,
    // evaluates the cheap bbox-IoU matrix, and only scans RLE runs for pairs
    // whose boxes overlap.  The first Rust port accidentally recomputed both
    // bboxes inside every pairwise exact-IoU call, making complexity depend on
    // (num_dt * num_gt * RLE_complexity).  Precompute all per-instance state
    // here so the expensive run scan is only paid for surviving pairs.
    let dt_bboxes: Vec<[f64; 4]> = dt_rles.iter().map(bbox_one).collect();
    let gt_bboxes: Vec<[f64; 4]> = gt_rles.iter().map(bbox_one).collect();
    let dt_areas: Vec<u32> = dt_rles.iter().map(area_one).collect();
    let m = dt_rles.len();
    let n = gt_rles.len();
    let mut values = Vec::<f64>::with_capacity(m * n);
    for d in 0..m {
        for g in 0..n {
            let crowd = iscrowd[g] != 0;
            let bbox_iou = bbox_iou_xywh(&dt_bboxes[d], &gt_bboxes[g], crowd);
            let value = if bbox_iou > 0.0 {
                rle_iou_runs(&dt_rles[d], &gt_rles[g], crowd, dt_areas[d])
            } else {
                0.0
            };
            values.push(value);
        }
    }
    let out = Array2::from_shape_vec((m, n), values)
        .map_err(|ex| value_error(format!("unable to shape IoU output: {ex}")))?;
    Ok(out.into_pyarray_bound(py).into_py(py))
}

#[pyfunction(name = "toBbox")]
pub fn to_bbox<'py>(
    py: Python<'py>,
    rle_objs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rles = rles_from_list(rle_objs)?;
    let mut out = Array2::<f64>::zeros((rles.len(), 4));
    for (i, rle) in rles.iter().enumerate() {
        let b = bbox_one(rle);
        for j in 0..4 {
            out[(i, j)] = b[j];
        }
    }
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction(name = "frBbox")]
pub fn fr_bbox(
    py: Python<'_>,
    bb: PyReadonlyArray2<'_, f64>,
    h: usize,
    w: usize,
) -> PyResult<Vec<PyObject>> {
    let arr = bb.as_array();
    if arr.ncols() != 4 {
        return Err(value_error(format!("bbox input must have shape Nx4, got {:?}", arr.shape())));
    }
    let mut result = Vec::with_capacity(arr.nrows());
    for i in 0..arr.nrows() {
        let b = arr.row(i);
        let xs = b[0];
        let ys = b[1];
        let xe = xs + b[2];
        let ye = ys + b[3];
        let poly = [xs, ys, xs, ye, xe, ye, xe, ys];
        result.push(rle_to_object(py, &polygon_to_rle(&poly, h, w))?);
    }
    Ok(result)
}

#[pyfunction(name = "frPoly")]
pub fn fr_poly(
    py: Python<'_>,
    poly: Vec<Vec<f64>>,
    h: usize,
    w: usize,
) -> PyResult<Vec<PyObject>> {
    let mut result = Vec::with_capacity(poly.len());
    for part in poly {
        if part.len() % 2 != 0 {
            return Err(value_error("polygon coordinate vectors must have even length"));
        }
        result.push(rle_to_object(py, &polygon_to_rle(&part, h, w))?);
    }
    Ok(result)
}

#[pyfunction(name = "frUncompressedRLE")]
pub fn fr_uncompressed_rle(
    py: Python<'_>,
    uc_rles: &Bound<'_, PyAny>,
    _h: usize,
    _w: usize,
) -> PyResult<Vec<PyObject>> {
    let list = uc_rles.downcast::<PyList>()?;
    let mut result = Vec::with_capacity(list.len());
    for item in list.iter() {
        let d = item.downcast::<PyDict>()?;
        let size_obj = d.get_item("size")?.ok_or_else(|| value_error("RLE missing size"))?;
        let counts_obj = d.get_item("counts")?.ok_or_else(|| value_error("RLE missing counts"))?;
        let size: Vec<usize> = size_obj.extract()?;
        let counts: Vec<u32> = counts_obj.extract()?;
        if size.len() != 2 {
            return Err(value_error("RLE size must have two values"));
        }
        let rle = Rle { h: size[0], w: size[1], counts };
        result.push(rle_to_object(py, &rle)?);
    }
    Ok(result)
}

#[pyfunction]
pub fn _rle_bytes_to_array<'py>(
    py: Python<'py>,
    counts: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let bytes = extract_count_bytes(counts)?;
    let values = bytes_to_counts(&bytes)?;
    Ok(Array1::from_vec(values).into_pyarray_bound(py))
}

#[pyfunction]
pub fn _rle_array_to_bytes<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyBytes>> {
    let view = counts.as_array();
    let values: Vec<u32> = view.iter().copied().collect();
    Ok(PyBytes::new_bound(py, &counts_to_bytes(&values)))
}
