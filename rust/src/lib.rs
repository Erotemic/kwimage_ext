use pyo3::prelude::*;

mod boxes;
mod mask;
mod nms;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn capabilities() -> Vec<&'static str> {
    vec![
        "boxes",
        "mask-rle",
        "mask-iou",
        "cpu-nms",
    ]
}

#[pymodule]
fn _rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(capabilities, m)?)?;

    m.add_function(wrap_pyfunction!(boxes::bbox_ious_c, m)?)?;
    m.add_function(wrap_pyfunction!(boxes::bbox_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(boxes::bbox_intersections, m)?)?;
    m.add_function(wrap_pyfunction!(boxes::anchor_intersections, m)?)?;
    m.add_function(wrap_pyfunction!(boxes::bbox_intersections_self, m)?)?;
    m.add_function(wrap_pyfunction!(boxes::bbox_similarities, m)?)?;

    m.add_function(wrap_pyfunction!(mask::encode, m)?)?;
    m.add_function(wrap_pyfunction!(mask::decode, m)?)?;
    m.add_function(wrap_pyfunction!(mask::merge, m)?)?;
    m.add_function(wrap_pyfunction!(mask::area, m)?)?;
    m.add_function(wrap_pyfunction!(mask::iou, m)?)?;
    m.add_function(wrap_pyfunction!(mask::to_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(mask::fr_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(mask::fr_poly, m)?)?;
    m.add_function(wrap_pyfunction!(mask::fr_uncompressed_rle, m)?)?;
    m.add_function(wrap_pyfunction!(mask::_rle_bytes_to_array, m)?)?;
    m.add_function(wrap_pyfunction!(mask::_rle_array_to_bytes, m)?)?;

    m.add_function(wrap_pyfunction!(nms::cpu_nms, m)?)?;
    Ok(())
}
