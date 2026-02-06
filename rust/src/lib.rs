use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

fn to_vec2_f32(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
    obj.extract::<Vec<Vec<f32>>>()
}

fn to_vec1_f32(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    obj.extract::<Vec<f32>>()
}

fn numpy_f32_2d(py: Python<'_>, data: Vec<f32>, n: usize, k: usize) -> PyResult<PyObject> {
    let np = py.import_bound("numpy")?;
    let arr = np.call_method1("array", (data,))?;
    let arr = arr.call_method1("astype", ("float32",))?;
    let arr = arr.call_method1("reshape", (n, k))?;
    Ok(arr.into_py(py))
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn bbox_ious_c(py: Python<'_>, boxes: &Bound<'_, PyAny>, query_boxes: &Bound<'_, PyAny>, bias: Option<f32>) -> PyResult<PyObject> {
    let boxes = to_vec2_f32(boxes)?;
    let q = to_vec2_f32(query_boxes)?;
    let b = bias.unwrap_or(1.0);
    let n = boxes.len();
    let k = q.len();
    let mut out = vec![0.0f32; n * k];
    for (qi, qb) in q.iter().enumerate() {
        let q_area = (qb[2] - qb[0] + b) * (qb[3] - qb[1] + b);
        for (bi, bb) in boxes.iter().enumerate() {
            let iw = bb[2].min(qb[2]) - bb[0].max(qb[0]) + b;
            if iw > 0.0 {
                let ih = bb[3].min(qb[3]) - bb[1].max(qb[1]) + b;
                if ih > 0.0 {
                    let box_area = (bb[2] - bb[0] + b) * (bb[3] - bb[1] + b);
                    let inter = iw * ih;
                    out[bi * k + qi] = inter / (q_area + box_area - inter);
                }
            }
        }
    }
    numpy_f32_2d(py, out, n, k)
}

#[pyfunction]
fn bbox_overlaps(py: Python<'_>, boxes: &Bound<'_, PyAny>, query_boxes: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    bbox_ious_c(py, boxes, query_boxes, Some(1.0))
}

#[pyfunction]
fn bbox_intersections(py: Python<'_>, boxes: &Bound<'_, PyAny>, query_boxes: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let boxes = to_vec2_f32(boxes)?;
    let q = to_vec2_f32(query_boxes)?;
    let n = boxes.len();
    let k = q.len();
    let mut out = vec![0.0f32; n * k];
    for (qi, qb) in q.iter().enumerate() {
        let q_area = (qb[2] - qb[0] + 1.0) * (qb[3] - qb[1] + 1.0);
        for (bi, bb) in boxes.iter().enumerate() {
            let iw = bb[2].min(qb[2]) - bb[0].max(qb[0]) + 1.0;
            if iw > 0.0 {
                let ih = bb[3].min(qb[3]) - bb[1].max(qb[1]) + 1.0;
                if ih > 0.0 {
                    out[bi * k + qi] = (iw * ih) / q_area;
                }
            }
        }
    }
    numpy_f32_2d(py, out, n, k)
}

#[pyfunction]
fn anchor_intersections(py: Python<'_>, anchors: &Bound<'_, PyAny>, query_boxes: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let anchors = to_vec2_f32(anchors)?;
    let q = to_vec2_f32(query_boxes)?;
    let n = anchors.len();
    let k = q.len();
    let mut out = vec![0.0f32; n * k];
    for (ai, a) in anchors.iter().enumerate() {
        let a_area = a[0] * a[1];
        for (qi, qb) in q.iter().enumerate() {
            let bw = qb[2] - qb[0] + 1.0;
            let bh = qb[3] - qb[1] + 1.0;
            let iw = a[0].min(bw);
            let ih = a[1].min(bh);
            let inter = iw * ih;
            out[ai * k + qi] = inter / (a_area + bw * bh - inter);
        }
    }
    numpy_f32_2d(py, out, n, k)
}

#[pyfunction]
fn bbox_intersections_self(py: Python<'_>, boxes: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let boxes = to_vec2_f32(boxes)?;
    let n = boxes.len();
    let mut out = vec![0.0f32; n * n];
    for k in 0..n {
        let bk = &boxes[k];
        let area = (bk[2] - bk[0] + 1.0) * (bk[3] - bk[1] + 1.0);
        for nn in (k + 1)..n {
            let b = &boxes[nn];
            let iw = b[2].min(bk[2]) - b[0].max(bk[0]) + 1.0;
            if iw > 0.0 {
                let ih = b[3].min(bk[3]) - b[1].max(bk[1]) + 1.0;
                if ih > 0.0 {
                    out[k * n + nn] = (iw * ih) / area;
                }
            }
        }
    }
    numpy_f32_2d(py, out, n, n)
}

#[pyfunction]
fn bbox_similarities(py: Python<'_>, boxes: &Bound<'_, PyAny>, query_boxes: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let boxes = to_vec2_f32(boxes)?;
    let q = to_vec2_f32(query_boxes)?;
    let n = boxes.len();
    let k = q.len();
    let mut out = vec![0.0f32; n * k];
    for (bi, b1) in boxes.iter().enumerate() {
        let cx1 = (b1[0] + b1[2]) * 0.5;
        let cy1 = (b1[1] + b1[3]) * 0.5;
        let w1 = b1[2] - b1[0] + 1.0;
        let h1 = b1[3] - b1[1] + 1.0;
        for (qi, b2) in q.iter().enumerate() {
            let cx2 = (b2[0] + b2[2]) * 0.5;
            let cy2 = (b2[1] + b2[3]) * 0.5;
            let w2 = b2[2] - b2[0] + 1.0;
            let h2 = b2[3] - b2[1] + 1.0;
            let loc_dist = (cx1 - cx2).abs() / (w1 + w2) + (cy1 - cy2).abs() / (h1 + h2);
            let shape_dist = ((w2 * h2) / (w1 * h1) - 1.0).abs();
            out[bi * k + qi] = -((loc_dist + 0.001).ln()) - shape_dist * shape_dist + 1.0;
        }
    }
    numpy_f32_2d(py, out, n, k)
}

#[pyfunction]
fn cpu_nms(ltrb: &Bound<'_, PyAny>, scores: &Bound<'_, PyAny>, thresh: f32, bias: Option<f32>) -> PyResult<Vec<usize>> {
    let boxes = to_vec2_f32(ltrb)?;
    let scores = to_vec1_f32(scores)?;
    let b = bias.unwrap_or(0.0);
    let n = boxes.len();
    if scores.len() != n {
        return Err(PyValueError::new_err("scores length must match boxes"));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap());
    let mut suppressed = vec![false; n];
    let mut keep: Vec<usize> = Vec::new();
    let areas: Vec<f32> = boxes.iter().map(|bb| (bb[2]-bb[0]+b)*(bb[3]-bb[1]+b)).collect();
    for oi in 0..n {
        let i = order[oi];
        if suppressed[i] { continue; }
        keep.push(i);
        for oj in (oi+1)..n {
            let j = order[oj];
            if suppressed[j] { continue; }
            let iw = boxes[i][2].min(boxes[j][2]) - boxes[i][0].max(boxes[j][0]) + b;
            if iw <= 0.0 { continue; }
            let ih = boxes[i][3].min(boxes[j][3]) - boxes[i][1].max(boxes[j][1]) + b;
            if ih <= 0.0 { continue; }
            let inter = iw * ih;
            let ovr = inter / (areas[i] + areas[j] - inter);
            if ovr > thresh { suppressed[j] = true; }
        }
    }
    Ok(keep)
}

#[pyfunction]
fn soft_nms(py: Python<'_>, ltrb: &Bound<'_, PyAny>, scores: &Bound<'_, PyAny>, thresh: Option<f32>, overlap_thresh: Option<f32>, sigma: Option<f32>, bias: Option<f32>, method: Option<u32>) -> PyResult<PyObject> {
    let mut boxes = to_vec2_f32(ltrb)?;
    let mut scores = to_vec1_f32(scores)?;
    let mut inds: Vec<usize> = (0..boxes.len()).collect();
    let mut n = boxes.len();
    let thresh = thresh.unwrap_or(0.001);
    let overlap_thresh = overlap_thresh.unwrap_or(0.3);
    let sigma = sigma.unwrap_or(0.5);
    let bias = bias.unwrap_or(0.0);
    let method = method.unwrap_or(0);

    for i in 0..n {
        let mut maxpos = i;
        let mut maxscore = scores[i];
        let mut pos = i + 1;
        while pos < n {
            if scores[pos] > maxscore { maxscore = scores[pos]; maxpos = pos; }
            pos += 1;
        }
        boxes.swap(i, maxpos);
        scores.swap(i, maxpos);
        inds.swap(i, maxpos);

        let tx1 = boxes[i][0]; let ty1 = boxes[i][1]; let tx2 = boxes[i][2]; let ty2 = boxes[i][3];
        pos = i + 1;
        while pos < n {
            let x1 = boxes[pos][0]; let y1 = boxes[pos][1]; let x2 = boxes[pos][2]; let y2 = boxes[pos][3];
            let area = (x2 - x1 + bias) * (y2 - y1 + bias);
            let iw = tx2.min(x2) - tx1.max(x1) + bias;
            if iw > 0.0 {
                let ih = ty2.min(y2) - ty1.max(y1) + bias;
                if ih > 0.0 {
                    let ua = (tx2 - tx1 + bias) * (ty2 - ty1 + bias) + area - iw * ih;
                    let ov = iw * ih / ua;
                    let weight = if method == 1 {
                        if ov > overlap_thresh { 1.0 - ov } else { 1.0 }
                    } else if method == 2 {
                        (-(ov * ov) / sigma).exp()
                    } else if ov > overlap_thresh { 0.0 } else { 1.0 };
                    scores[pos] *= weight;
                    if scores[pos] < thresh {
                        boxes.swap(pos, n - 1);
                        scores.swap(pos, n - 1);
                        inds.swap(pos, n - 1);
                        n -= 1;
                        if pos == 0 { break; }
                        pos -= 1;
                    }
                }
            }
            pos += 1;
        }
    }

    let np = py.import_bound("numpy")?;
    let out = np.call_method1("array", (inds[..n].to_vec(),))?;
    Ok(out.into_py(py))
}

fn mask_mod(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import_bound("pycocotools.mask")
}

#[pyfunction]
fn encode(py: Python<'_>, mask: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    Ok(mask_mod(py)?.getattr("encode")?.call1((mask,))?.into_py(py))
}
#[pyfunction]
fn decode(py: Python<'_>, rle: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    Ok(mask_mod(py)?.getattr("decode")?.call1((rle,))?.into_py(py))
}
#[pyfunction]
fn merge(py: Python<'_>, rles: &Bound<'_, PyAny>, intersect: Option<bool>) -> PyResult<PyObject> {
    Ok(mask_mod(py)?.getattr("merge")?.call1((rles, intersect.unwrap_or(false)))?.into_py(py))
}
#[pyfunction]
fn area(py: Python<'_>, rles: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    Ok(mask_mod(py)?.getattr("area")?.call1((rles,))?.into_py(py))
}
#[pyfunction]
fn iou(py: Python<'_>, dt: &Bound<'_, PyAny>, gt: &Bound<'_, PyAny>, iscrowd: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    Ok(mask_mod(py)?.getattr("iou")?.call1((dt, gt, iscrowd))?.into_py(py))
}
#[pyfunction]
#[pyo3(name = "toBbox")]
fn to_bbox(py: Python<'_>, rles: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    Ok(mask_mod(py)?.getattr("toBbox")?.call1((rles,))?.into_py(py))
}
#[pyfunction]
#[pyo3(name = "frBbox")]
fn fr_bbox(py: Python<'_>, bb: &Bound<'_, PyAny>, h: u32, w: u32) -> PyResult<PyObject> {
    Ok(mask_mod(py)?.getattr("frBbox")?.call1((bb, h, w))?.into_py(py))
}
#[pyfunction]
#[pyo3(name = "frPoly")]
fn fr_poly(py: Python<'_>, p: &Bound<'_, PyAny>, h: u32, w: u32) -> PyResult<PyObject> {
    Ok(mask_mod(py)?.getattr("frPoly")?.call1((p, h, w))?.into_py(py))
}
#[pyfunction]
#[pyo3(name = "frUncompressedRLE")]
fn fr_uncompressed_rle(py: Python<'_>, uc: &Bound<'_, PyAny>, h: u32, w: u32) -> PyResult<PyObject> {
    Ok(mask_mod(py)?.getattr("frUncompressedRLE")?.call1((uc, h, w))?.into_py(py))
}
#[pyfunction]
#[pyo3(name = "frPyObjects")]
fn fr_py_objects(py: Python<'_>, pyobj: &Bound<'_, PyAny>, h: u32, w: u32) -> PyResult<PyObject> {
    Ok(mask_mod(py)?.getattr("frPyObjects")?.call1((pyobj, h, w))?.into_py(py))
}

#[pyfunction]
fn _rle_bytes_to_array(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let np = py.import_bound("numpy")?;
    let arr = np.call_method1("frombuffer", (data, "uint8"))?;
    Ok(arr.into_py(py))
}

#[pyfunction]
fn _rle_array_to_bytes(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    Ok(data.call_method0("tobytes")?.into_py(py))
}

#[pymodule]
fn _rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(bbox_ious_c, m)?)?;
    m.add_function(wrap_pyfunction!(bbox_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(bbox_intersections, m)?)?;
    m.add_function(wrap_pyfunction!(anchor_intersections, m)?)?;
    m.add_function(wrap_pyfunction!(bbox_intersections_self, m)?)?;
    m.add_function(wrap_pyfunction!(bbox_similarities, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_nms, m)?)?;
    m.add_function(wrap_pyfunction!(soft_nms, m)?)?;
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(merge, m)?)?;
    m.add_function(wrap_pyfunction!(area, m)?)?;
    m.add_function(wrap_pyfunction!(iou, m)?)?;
    m.add_function(wrap_pyfunction!(to_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(fr_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(fr_poly, m)?)?;
    m.add_function(wrap_pyfunction!(fr_uncompressed_rle, m)?)?;
    m.add_function(wrap_pyfunction!(fr_py_objects, m)?)?;
    m.add_function(wrap_pyfunction!(_rle_bytes_to_array, m)?)?;
    m.add_function(wrap_pyfunction!(_rle_array_to_bytes, m)?)?;
    Ok(())
}
