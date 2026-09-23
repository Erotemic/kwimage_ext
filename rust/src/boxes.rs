use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use numpy::ndarray::Array2;
use pyo3::prelude::*;


#[inline(always)]
fn cython_builtin_min(a: f32, b: f32) -> f32 {
    // Cython's optimized two-argument Python builtin min() keeps the first
    // argument unless the second compares smaller.  That distinction is
    // observable for NaNs (and equal signed zeroes), so preserve it while
    // avoiding f32::min's different NaN-selection semantics.
    if b < a { b } else { a }
}

#[inline(always)]
fn cython_builtin_max(a: f32, b: f32) -> f32 {
    // Likewise, max(a, b) starts from a and only replaces it when b > a.
    if b > a { b } else { a }
}

#[derive(Clone, Copy)]
struct PreparedBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    area: f32,
}

fn check_boxes(name: &str, arr: &numpy::ndarray::ArrayView2<'_, f32>, width: usize) -> PyResult<()> {
    if arr.ncols() != width {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must have shape Nx{width}, got {:?}", arr.shape()
        )));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (boxes, query_boxes, bias=1.0))]
pub fn bbox_ious_c<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'py, f32>,
    query_boxes: PyReadonlyArray2<'py, f32>,
    bias: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let boxes = boxes.as_array();
    let query = query_boxes.as_array();
    check_boxes("boxes", &boxes, 4)?;
    check_boxes("query_boxes", &query, 4)?;
    let n = boxes.nrows();
    let k = query.nrows();
    let mut out = Array2::<f32>::zeros((n, k));

    // The overwhelmingly common case is a standard C-order Nx4 NumPy array.
    // ndarray's general indexing path must retain dynamic stride and bounds
    // checks, which are measurable when this loop executes millions of times.
    // Keep the general strided implementation below, but use flat slices for
    // standard-layout inputs and write each output row contiguously.
    if boxes.is_standard_layout() && query.is_standard_layout() {
        let boxes_data = boxes
            .as_slice()
            .expect("standard-layout boxes must expose a contiguous slice");
        let query_data = query
            .as_slice()
            .expect("standard-layout query boxes must expose a contiguous slice");
        let out_data = out
            .as_slice_mut()
            .expect("owned output array must use standard layout");

        // Prepack the query geometry once. Iterating this slice in lockstep
        // with each contiguous output row gives LLVM simple, equal-length
        // iterators instead of repeated q*4 arithmetic and bounds checks.
        let prepared_query: Vec<PreparedBox> = query_data
            .chunks_exact(4)
            .map(|q| PreparedBox {
                x1: q[0],
                y1: q[1],
                x2: q[2],
                y2: q[3],
                area: (q[2] - q[0] + bias) * (q[3] - q[1] + bias),
            })
            .collect();

        for (bi, b) in boxes_data.chunks_exact(4).enumerate() {
            let bx1 = b[0];
            let by1 = b[1];
            let bx2 = b[2];
            let by2 = b[3];
            let out_row = &mut out_data[(bi * k)..((bi + 1) * k)];

            // Preserve the historical Cython behavior for malformed boxes:
            // a NaN in any candidate-box coordinate leaves the pre-zeroed
            // overlap row untouched. Query-side NaNs are intentionally *not*
            // handled here; legacy Cython propagates those to NaN instead.
            if bx1.is_nan() || by1.is_nan() || bx2.is_nan() || by2.is_nan() {
                continue;
            }

            let barea = (bx2 - bx1 + bias) * (by2 - by1 + bias);
            for (cell, q) in out_row.iter_mut().zip(prepared_query.iter()) {
                let iw = cython_builtin_min(bx2, q.x2) - cython_builtin_max(bx1, q.x1) + bias;
                if iw > 0.0 {
                    let ih = cython_builtin_min(by2, q.y2) - cython_builtin_max(by1, q.y1) + bias;
                    if ih > 0.0 {
                        let inter = iw * ih;
                        *cell = inter / (q.area + barea - inter);
                    }
                }
            }
        }
    } else {
        // Preserve support for arbitrary NumPy strided views.
        for qi in 0..k {
            let q = query.row(qi);
            let qarea = (q[2] - q[0] + bias) * (q[3] - q[1] + bias);
            for bi in 0..n {
                let b = boxes.row(bi);
                if b[0].is_nan() || b[1].is_nan() || b[2].is_nan() || b[3].is_nan() {
                    continue;
                }
                let iw = cython_builtin_min(b[2], q[2]) - cython_builtin_max(b[0], q[0]) + bias;
                if iw > 0.0 {
                    let ih = cython_builtin_min(b[3], q[3]) - cython_builtin_max(b[1], q[1]) + bias;
                    if ih > 0.0 {
                        let barea = (b[2] - b[0] + bias) * (b[3] - b[1] + bias);
                        let inter = iw * ih;
                        out[(bi, qi)] = inter / (qarea + barea - inter);
                    }
                }
            }
        }
    }
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
pub fn bbox_overlaps<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'py, f32>,
    query_boxes: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    bbox_ious_c(py, boxes, query_boxes, 1.0)
}

#[pyfunction]
pub fn bbox_intersections<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'py, f32>,
    query_boxes: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let boxes = boxes.as_array();
    let query = query_boxes.as_array();
    check_boxes("boxes", &boxes, 4)?;
    check_boxes("query_boxes", &query, 4)?;
    let n = boxes.nrows();
    let k = query.nrows();
    let mut out = Array2::<f32>::zeros((n, k));
    for qi in 0..k {
        let q = query.row(qi);
        let qarea = (q[2] - q[0] + 1.0) * (q[3] - q[1] + 1.0);
        for bi in 0..n {
            let b = boxes.row(bi);
            let iw = b[2].min(q[2]) - b[0].max(q[0]) + 1.0;
            if iw > 0.0 {
                let ih = b[3].min(q[3]) - b[1].max(q[1]) + 1.0;
                if ih > 0.0 && qarea != 0.0 {
                    out[(bi, qi)] = iw * ih / qarea;
                }
            }
        }
    }
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
pub fn anchor_intersections<'py>(
    py: Python<'py>,
    anchors: PyReadonlyArray2<'py, f32>,
    query_boxes: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let anchors = anchors.as_array();
    let query = query_boxes.as_array();
    check_boxes("anchors", &anchors, 2)?;
    check_boxes("query_boxes", &query, 4)?;
    let n = anchors.nrows();
    let k = query.nrows();
    let mut out = Array2::<f32>::zeros((n, k));
    for ai in 0..n {
        let a = anchors.row(ai);
        let anchor_area = a[0] * a[1];
        for qi in 0..k {
            let q = query.row(qi);
            let boxw = q[2] - q[0] + 1.0;
            let boxh = q[3] - q[1] + 1.0;
            let inter = a[0].min(boxw) * a[1].min(boxh);
            let denom = anchor_area + boxw * boxh - inter;
            if denom != 0.0 {
                out[(ai, qi)] = inter / denom;
            }
        }
    }
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
pub fn bbox_intersections_self<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let boxes = boxes.as_array();
    check_boxes("boxes", &boxes, 4)?;
    let n = boxes.nrows();
    let mut out = Array2::<f32>::zeros((n, n));
    for k in 0..n {
        let q = boxes.row(k);
        let qarea = (q[2] - q[0] + 1.0) * (q[3] - q[1] + 1.0);
        for bi in (k + 1)..n {
            let b = boxes.row(bi);
            let iw = b[2].min(q[2]) - b[0].max(q[0]) + 1.0;
            if iw > 0.0 {
                let ih = b[3].min(q[3]) - b[1].max(q[1]) + 1.0;
                if ih > 0.0 && qarea != 0.0 {
                    out[(k, bi)] = iw * ih / qarea;
                }
            }
        }
    }
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
pub fn bbox_similarities<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'py, f32>,
    query_boxes: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let boxes = boxes.as_array();
    let query = query_boxes.as_array();
    check_boxes("boxes", &boxes, 4)?;
    check_boxes("query_boxes", &query, 4)?;
    let n = boxes.nrows();
    let k = query.nrows();
    let mut out = Array2::<f32>::zeros((n, k));
    for bi in 0..n {
        let b = boxes.row(bi);
        let cx1 = (b[0] + b[2]) * 0.5;
        let cy1 = (b[1] + b[3]) * 0.5;
        let w1 = b[2] - b[0] + 1.0;
        let h1 = b[3] - b[1] + 1.0;
        for qi in 0..k {
            let q = query.row(qi);
            let cx2 = (q[0] + q[2]) * 0.5;
            let cy2 = (q[1] + q[3]) * 0.5;
            let w2 = q[2] - q[0] + 1.0;
            let h2 = q[3] - q[1] + 1.0;
            let loc_dist = (cx1 - cx2).abs() / (w1 + w2) + (cy1 - cy2).abs() / (h1 + h2);
            let shape_dist = (w2 * h2 / (w1 * h1) - 1.0).abs();
            out[(bi, qi)] = -(loc_dist + 0.001).ln() - shape_dist * shape_dist + 1.0;
        }
    }
    Ok(out.into_pyarray_bound(py))
}
