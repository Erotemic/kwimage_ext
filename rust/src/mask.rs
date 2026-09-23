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

#[inline]
fn for_each_compressed_count(
    bytes: &[u8],
    mut visit: impl FnMut(usize, u32),
) -> PyResult<()> {
    // Keep the compressed-RLE decoder as one tight monomorphized loop.  A
    // previous stateful `next_count()` abstraction made fragmented RLEs pay a
    // PyResult<Option<_>> round trip for every run; the claim-ready benchmark
    // showed that cost in decode/area/merge.  Callers supply an inlined visitor
    // so they can either materialize counts or consume them directly.
    let mut p = 0usize;
    let mut index = 0usize;
    let mut previous_same_parity = [0u32; 2];
    while p < bytes.len() && bytes[p] != 0 {
        let byte = bytes[p];
        if !(48..=111).contains(&byte) {
            return Err(value_error("invalid compressed RLE character"));
        }
        let first = (byte - 48) as i64;
        let mut x = first & 0x1f;
        p += 1;
        if first & 0x20 == 0 {
            // Most fragmented-mask counts fit in one byte. Avoid entering the
            // general variable-length loop for this overwhelmingly common
            // case while preserving signed differential decoding.
            if first & 0x10 != 0 {
                x |= -1i64 << 5;
            }
        } else {
            let mut k = 1usize;
            loop {
                if p >= bytes.len() {
                    return Err(value_error("truncated compressed RLE counts"));
                }
                let byte = bytes[p];
                if !(48..=111).contains(&byte) {
                    return Err(value_error("invalid compressed RLE character"));
                }
                let c = (byte - 48) as i64;
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
        }

        let parity = index & 1;
        if index > 2 {
            x += previous_same_parity[parity] as i64;
        }
        if x < 0 || x > u32::MAX as i64 {
            return Err(value_error("compressed RLE decoded outside uint32 range"));
        }
        let count = x as u32;
        previous_same_parity[parity] = count;
        visit(index, count);
        index += 1;
    }
    Ok(())
}


struct CompressedCountCursor<'a> {
    bytes: &'a [u8],
    p: usize,
    index: usize,
    previous_same_parity: [u32; 2],
    error: Option<&'static str>,
}

impl<'a> CompressedCountCursor<'a> {
    #[inline]
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            p: 0,
            index: 0,
            previous_same_parity: [0; 2],
            error: None,
        }
    }

    #[inline(always)]
    fn next_count(&mut self) -> Option<u32> {
        // A decode error is terminal: the caller treats this `None` exactly
        // like end-of-stream and calls `finish()` before returning.  It never
        // invokes the cursor again, so checking `error` here would add a
        // redundant branch to every successfully decoded count.
        if self.p >= self.bytes.len() || self.bytes[self.p] == 0 {
            return None;
        }

        let byte = self.bytes[self.p];
        if !(48..=111).contains(&byte) {
            self.error = Some("invalid compressed RLE character");
            return None;
        }
        let first = (byte - 48) as i64;
        let mut x = first & 0x1f;
        self.p += 1;
        if first & 0x20 == 0 {
            if first & 0x10 != 0 {
                x |= -1i64 << 5;
            }
        } else {
            let mut k = 1usize;
            loop {
                if self.p >= self.bytes.len() {
                    self.error = Some("truncated compressed RLE counts");
                    return None;
                }
                let byte = self.bytes[self.p];
                if !(48..=111).contains(&byte) {
                    self.error = Some("invalid compressed RLE character");
                    return None;
                }
                let c = (byte - 48) as i64;
                x |= (c & 0x1f) << (5 * k);
                let more = c & 0x20;
                self.p += 1;
                k += 1;
                if more == 0 {
                    if c & 0x10 != 0 {
                        x |= -1i64 << (5 * k);
                    }
                    break;
                }
            }
        }

        let parity = self.index & 1;
        if self.index > 2 {
            x += self.previous_same_parity[parity] as i64;
        }
        if x < 0 || x > u32::MAX as i64 {
            self.error = Some("compressed RLE decoded outside uint32 range");
            return None;
        }

        let count = x as u32;
        self.previous_same_parity[parity] = count;
        self.index += 1;
        Some(count)
    }

    #[inline]
    fn finish(self) -> PyResult<()> {
        match self.error {
            Some(message) => Err(value_error(message)),
            None => Ok(()),
        }
    }
}

struct CountIntervalCursor<'a> {
    counts: &'a [u32],
    index: usize,
    position: u32,
}

impl<'a> CountIntervalCursor<'a> {
    #[inline]
    fn new(counts: &'a [u32]) -> Self {
        Self { counts, index: 0, position: 0 }
    }

    #[inline(always)]
    fn next_interval(&mut self) -> Option<(u32, u32)> {
        while self.index < self.counts.len() {
            self.position = self.position.wrapping_add(self.counts[self.index]);
            self.index += 1;
            if self.index == self.counts.len() {
                return None;
            }
            let start = self.position;
            let foreground = self.counts[self.index];
            self.position = self.position.wrapping_add(foreground);
            self.index += 1;
            if foreground != 0 {
                return Some((start, self.position));
            }
        }
        None
    }
}

struct CompressedIntervalCursor<'a> {
    counts: CompressedCountCursor<'a>,
    position: u32,
}

impl<'a> CompressedIntervalCursor<'a> {
    #[inline]
    fn new(bytes: &'a [u8]) -> Self {
        Self { counts: CompressedCountCursor::new(bytes), position: 0 }
    }

    #[inline(always)]
    fn next_interval(&mut self) -> Option<(u32, u32)> {
        loop {
            let background = self.counts.next_count()?;
            self.position = self.position.wrapping_add(background);
            let foreground = self.counts.next_count()?;
            let start = self.position;
            self.position = self.position.wrapping_add(foreground);
            if foreground != 0 {
                return Some((start, self.position));
            }
        }
    }

    #[inline]
    fn finish(self) -> PyResult<()> {
        self.counts.finish()
    }
}

fn decode_counts_into(bytes: &[u8], counts: &mut Vec<u32>) -> PyResult<()> {
    counts.clear();
    if counts.capacity() < bytes.len() {
        // One decoded count consumes at least one compressed byte, so this is
        // an exact upper bound and prevents growth on fragmented masks.
        counts.reserve(bytes.len());
    }
    for_each_compressed_count(bytes, |_, count| counts.push(count))
}

fn bytes_to_counts(bytes: &[u8]) -> PyResult<Vec<u32>> {
    let mut counts = Vec::with_capacity(bytes.len());
    decode_counts_into(bytes, &mut counts)?;
    Ok(counts)
}

fn rle_bytes_to_object(py: Python<'_>, h: usize, w: usize, counts: &[u8]) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("size", vec![h, w])?;
    d.set_item("counts", PyBytes::new_bound(py, counts))?;
    Ok(d.into_py(py))
}

fn rle_to_object(py: Python<'_>, rle: &Rle) -> PyResult<PyObject> {
    let counts = counts_to_bytes(&rle.counts);
    rle_bytes_to_object(py, rle.h, rle.w, &counts)
}

fn with_count_bytes<T>(
    obj: &Bound<'_, PyAny>,
    f: impl FnOnce(&[u8]) -> PyResult<T>,
) -> PyResult<T> {
    if let Ok(b) = obj.downcast::<PyBytes>() {
        return f(b.as_bytes());
    }
    if let Ok(s) = obj.extract::<String>() {
        return f(s.as_bytes());
    }
    Err(value_error("compressed RLE counts must be bytes or str"))
}

fn rle_dimensions(d: &Bound<'_, PyDict>) -> PyResult<(usize, usize)> {
    let size_obj = d.get_item("size")?.ok_or_else(|| value_error("RLE missing size"))?;

    // `encode()` and pycocotools both use an ordinary two-element Python
    // list. Avoid allocating a temporary Vec for that overwhelmingly common
    // path while retaining the old generic sequence extraction as fallback.
    if let Ok(size) = size_obj.downcast::<PyList>() {
        if size.len() != 2 {
            return Err(value_error("RLE size must contain [height, width]"));
        }
        return Ok((size.get_item(0)?.extract()?, size.get_item(1)?.extract()?));
    }

    let size: Vec<usize> = size_obj.extract()?;
    if size.len() != 2 {
        return Err(value_error("RLE size must contain [height, width]"));
    }
    Ok((size[0], size[1]))
}

fn rle_from_object(obj: &Bound<'_, PyAny>) -> PyResult<Rle> {
    let d = obj.downcast::<PyDict>()?;
    let counts_obj = d.get_item("counts")?.ok_or_else(|| value_error("RLE missing counts"))?;
    let (h, w) = rle_dimensions(d)?;
    let counts = with_count_bytes(&counts_obj, bytes_to_counts)?;
    Ok(Rle { h, w, counts })
}

fn rles_from_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Rle>> {
    let list = obj.downcast::<PyList>()?;
    list.iter().map(|item| rle_from_object(&item)).collect()
}

fn area_one(rle: &Rle) -> u32 {
    rle.counts.iter().skip(1).step_by(2).copied().sum()
}

#[inline]
fn compressed_area(bytes: &[u8]) -> PyResult<u32> {
    // `area` needs only odd-indexed foreground runs. Consume the compressed
    // stream directly, without allocating a run vector, while sharing the same
    // tight decoder used by materializing operations.
    let mut area = 0u32;
    for_each_compressed_count(bytes, |index, count| {
        if index & 1 != 0 {
            area = area.wrapping_add(count);
        }
    })?;
    Ok(area)
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

#[inline(always)]
fn push_compressed_count(
    out: &mut Vec<u8>,
    count: u32,
    count_index: usize,
    previous_same_parity: &mut [u32; 2],
) {
    // COCO's compressed RLE stores count[i] - count[i - 2] from the fourth
    // run onward. A two-entry parity ring is therefore enough to compress a
    // run as soon as it is discovered; no intermediate Vec<u32> is needed.
    let parity = count_index & 1;
    let mut x = count as i64;
    if count_index > 2 {
        x -= previous_same_parity[parity] as i64;
    }
    previous_same_parity[parity] = count;

    // The common case for random/sparse masks is a small same-parity delta.
    // Values in [-16, 15] fit in exactly one COCO compressed byte. Avoid the
    // general signed-LEB-style loop for that hot path.
    if (-16..=15).contains(&x) {
        out.push(((x & 0x1f) as u8) + 48);
        return;
    }

    loop {
        let mut c = (x & 0x1f) as u8;
        x >>= 5;
        let more = if c & 0x10 != 0 { x != -1 } else { x != 0 };
        if more {
            c |= 0x20;
        }
        out.push(c + 48);
        if !more {
            break;
        }
    }
}

#[inline(always)]
fn word_has_zero_byte(word: u64) -> bool {
    // Portable scalar zero-byte detection. This is the classic subtract / mask
    // trick and deliberately uses only baseline integer operations: no target
    // features, runtime SIMD dispatch, or unsafe loads are required.
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    (word.wrapping_sub(LO) & !word & HI) != 0
}

#[inline(always)]
fn word_matches_logical_value(word: u64, logical_value: u8) -> bool {
    if logical_value == 0 {
        word == 0
    } else {
        // Rust's mask encoder treats every non-zero byte as foreground.  A
        // word is therefore entirely foreground exactly when it contains no
        // zero byte.
        !word_has_zero_byte(word)
    }
}

#[inline]
fn find_logical_transition(data: &[u8], start: usize, logical_value: u8) -> usize {
    let mut pos = start;

    // Structured masks are dominated by long uniform runs. Check 32 bytes at
    // a time using four portable u64 loads, then narrow to one word and finally
    // individual bytes near the transition. `from_ne_bytes` copies from the
    // slice safely and lets LLVM turn these into ordinary unaligned loads on
    // targets where that is profitable.
    while data.len() - pos >= 32 {
        let chunk = &data[pos..pos + 32];
        let w0 = u64::from_ne_bytes(chunk[0..8].try_into().unwrap());
        let w1 = u64::from_ne_bytes(chunk[8..16].try_into().unwrap());
        let w2 = u64::from_ne_bytes(chunk[16..24].try_into().unwrap());
        let w3 = u64::from_ne_bytes(chunk[24..32].try_into().unwrap());
        if word_matches_logical_value(w0, logical_value)
            && word_matches_logical_value(w1, logical_value)
            && word_matches_logical_value(w2, logical_value)
            && word_matches_logical_value(w3, logical_value)
        {
            pos += 32;
        } else {
            break;
        }
    }

    while data.len() - pos >= 8 {
        let word = u64::from_ne_bytes(data[pos..pos + 8].try_into().unwrap());
        if word_matches_logical_value(word, logical_value) {
            pos += 8;
        } else {
            break;
        }
    }

    while pos < data.len() && u8::from(data[pos] != 0) == logical_value {
        pos += 1;
    }
    pos
}

fn prefer_long_run_encode(data: &[u8]) -> bool {
    // The ordinary fused encoder wins when transitions are common because it
    // avoids an intermediate run-count vector and second compression pass.
    // Structured masks spend almost all of their time scanning long runs, so
    // sample four windows and use a word-at-a-time scanner when transition
    // density is very low. A wrong prediction affects only performance; both
    // encoders produce the same canonical COCO byte stream.
    if data.len() <= 1 {
        return true;
    }

    const SAMPLE_WINDOW: usize = 256;
    const SAMPLE_WINDOWS: usize = 4;
    const LOW_TRANSITION_DENOMINATOR: usize = 32;

    let window_len = data.len().min(SAMPLE_WINDOW);
    let num_windows = if data.len() <= window_len { 1 } else { SAMPLE_WINDOWS };
    let span = data.len() - window_len;
    let total_comparisons = num_windows * (window_len - 1);
    let max_long_run_transitions = total_comparisons / LOW_TRANSITION_DENOMINATOR;
    let mut transitions = 0usize;

    for window_index in 0..num_windows {
        let start = if num_windows == 1 {
            0
        } else {
            window_index * span / (num_windows - 1)
        };
        let sample = &data[start..(start + window_len)];
        let mut iter = sample.iter();
        let Some(&first) = iter.next() else {
            continue;
        };
        let mut previous = first != 0;
        for &pixel in iter {
            let value = pixel != 0;
            if value != previous {
                transitions += 1;
                // This is an exact early reject, not a heuristic change. Once
                // the final threshold has been exceeded, no remaining sample
                // can make the classifier return true.
                if transitions > max_long_run_transitions {
                    return false;
                }
            }
            previous = value;
        }
    }

    // <= 1 transition per 32 sampled adjacencies is strongly characteristic
    // of large structured runs, while sparse-random and dense-random masks are
    // comfortably above this threshold.  A wrong prediction only affects
    // performance; both paths have identical output semantics.
    true
}

fn encode_plane_bytes_long_runs(h: usize, w: usize, data: &[u8]) -> Vec<u8> {
    debug_assert_eq!(data.len(), h * w);

    // Long-run masks have very few encoded counts, so a small reservation is
    // sufficient. The hot work is finding transitions, which is accelerated by
    // `find_logical_transition` without changing the portable wheel ISA.
    let mut out = Vec::with_capacity(64);
    let mut position = 0usize;
    let mut logical_value = 0u8;
    let mut count_index = 0usize;
    let mut previous_same_parity = [0u32; 2];

    if data.is_empty() {
        push_compressed_count(
            &mut out,
            0,
            count_index,
            &mut previous_same_parity,
        );
        return out;
    }

    while position < data.len() {
        let transition = find_logical_transition(data, position, logical_value);
        let run_length = (transition - position) as u32;
        push_compressed_count(
            &mut out,
            run_length,
            count_index,
            &mut previous_same_parity,
        );
        count_index += 1;
        position = transition;
        logical_value ^= 1;
    }
    out
}

fn encode_plane_bytes_memory_order(h: usize, w: usize, data: &[u8]) -> Vec<u8> {
    debug_assert_eq!(data.len(), h * w);

    // Sparse masks commonly compress to well below one byte per pixel. This
    // initial reservation is deliberately modest; avoiding the entire
    // intermediate run-count allocation/pass matters more than exact sizing.
    let mut out = Vec::with_capacity(data.len() / 8 + 16);
    let mut previous = 0u8;
    let mut count = 0u32;
    let mut count_index = 0usize;
    let mut previous_same_parity = [0u32; 2];

    for &pixel in data {
        let value = if pixel == 0 { 0 } else { 1 };
        if value != previous {
            push_compressed_count(
                &mut out,
                count,
                count_index,
                &mut previous_same_parity,
            );
            count_index += 1;
            count = 0;
            previous = value;
        }
        count += 1;
    }
    push_compressed_count(
        &mut out,
        count,
        count_index,
        &mut previous_same_parity,
    );
    out
}

fn merge_counts_with_compressed(
    a: &[u32],
    b_bytes: &[u8],
    intersect: bool,
    out: &mut Vec<u32>,
) -> PyResult<()> {
    out.clear();
    // A decoded count consumes at least one compressed byte, so this remains
    // an upper bound without first materializing the right-hand RLE.
    let required_capacity = a.len().saturating_add(b_bytes.len());
    if out.capacity() < required_capacity {
        out.reserve(required_capacity);
    }

    let a_len = a.len();
    let mut ia = 1usize;
    let mut ca = a.first().copied().unwrap_or(0);

    // Stream the compressed RHS directly into the merge.  Round 2 decoded it
    // into a temporary Vec first; perf attributed 14% of structured merge and
    // 26% of fragmented merge to that extra pass.  This cursor defers error
    // conversion until the end so the hot loop does not carry PyResult per run.
    let mut b_decoder = CompressedCountCursor::new(b_bytes);
    let first_b = b_decoder.next_count();
    let mut cb = first_b.unwrap_or(0);
    let mut b_done = first_b.is_none();

    let mut va = false;
    let mut vb = false;
    let mut out_value = false;
    let mut run = 0u32;

    loop {
        let c = if ca < cb { ca } else { cb };
        // COCO's reference implementation accumulates unsigned counts with
        // normal uint arithmetic. wrapping_add matches that behavior without
        // the extra saturation logic in the previous Rust loop.
        run = run.wrapping_add(c);
        ca -= c;
        cb -= c;

        if ca == 0 && ia < a_len {
            ca = a[ia];
            ia += 1;
            va = !va;
        }
        if cb == 0 && !b_done {
            match b_decoder.next_count() {
                Some(next) => {
                    cb = next;
                    vb = !vb;
                }
                None => b_done = true,
            }
        }

        let next_value = if intersect { va && vb } else { va || vb };
        let done = ca == 0 && cb == 0 && ia >= a_len && b_done;
        if next_value != out_value || done {
            out.push(run);
            run = 0;
            out_value = next_value;
        }
        if done {
            break;
        }
    }

    b_decoder.finish()
}

fn union_counts_with_compressed(
    a: &[u32],
    b_bytes: &[u8],
    total: u32,
    out: &mut Vec<u32>,
) -> PyResult<()> {
    out.clear();
    let required_capacity = a.len().saturating_add(b_bytes.len());
    if out.capacity() < required_capacity {
        out.reserve(required_capacity);
    }

    // Treat each RLE as a sorted stream of foreground intervals.  The usual
    // run-boundary state machine handles every background and foreground run
    // separately; interval union handles one logical object per pair and can
    // coalesce overlap immediately as the accumulated union grows denser.
    let mut a_cursor = CountIntervalCursor::new(a);
    let mut b_cursor = CompressedIntervalCursor::new(b_bytes);
    let mut next_a = a_cursor.next_interval();
    let mut next_b = b_cursor.next_interval();
    let mut pending: Option<(u32, u32)> = None;
    let mut output_position = 0u32;

    while next_a.is_some() || next_b.is_some() {
        let interval = match (next_a, next_b) {
            (Some(a_interval), Some(b_interval)) => {
                if a_interval.0 <= b_interval.0 {
                    next_a = a_cursor.next_interval();
                    a_interval
                } else {
                    next_b = b_cursor.next_interval();
                    b_interval
                }
            }
            (Some(a_interval), None) => {
                next_a = a_cursor.next_interval();
                a_interval
            }
            (None, Some(b_interval)) => {
                next_b = b_cursor.next_interval();
                b_interval
            }
            (None, None) => unreachable!(),
        };

        match pending {
            Some((start, end)) if interval.0 <= end => {
                pending = Some((start, end.max(interval.1)));
            }
            Some((start, end)) => {
                out.push(start.wrapping_sub(output_position));
                out.push(end.wrapping_sub(start));
                output_position = end;
                pending = Some(interval);
            }
            None => pending = Some(interval),
        }
    }

    if let Some((start, end)) = pending {
        out.push(start.wrapping_sub(output_position));
        out.push(end.wrapping_sub(start));
        output_position = end;
    }
    if output_position < total || out.is_empty() {
        out.push(total.wrapping_sub(output_position));
    }
    b_cursor.finish()
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
            let plane = &data[start..stop];
            let counts = if prefer_long_run_encode(plane) {
                encode_plane_bytes_long_runs(h, w, plane)
            } else {
                encode_plane_bytes_memory_order(h, w, plane)
            };
            result.push(rle_bytes_to_object(py, h, w, &counts)?);
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
    let list = rle_objs.downcast::<PyList>()?;
    if list.len() == 0 {
        return rle_to_object(py, &Rle { h: 0, w: 0, counts: Vec::new() });
    }

    let first = list.get_item(0)?;
    let first_dict = first.downcast::<PyDict>()?;
    let (h, w) = rle_dimensions(first_dict)?;
    let first_counts_obj = first_dict
        .get_item("counts")?
        .ok_or_else(|| value_error("RLE missing counts"))?;
    let mut acc = with_count_bytes(&first_counts_obj, bytes_to_counts)?;
    let total = h.saturating_mul(w) as u32;

    // Reuse the accumulator/output buffers through the reduction while
    // streaming each compressed right-hand operand directly into the merge.
    let mut scratch = Vec::<u32>::new();
    for item in list.iter().skip(1) {
        let d = item.downcast::<PyDict>()?;
        let (other_h, other_w) = rle_dimensions(d)?;
        if other_h != h || other_w != w {
            return rle_to_object(py, &Rle { h: 0, w: 0, counts: Vec::new() });
        }
        let counts_obj = d
            .get_item("counts")?
            .ok_or_else(|| value_error("RLE missing counts"))?;
        if intersect == 0 {
            with_count_bytes(&counts_obj, |bytes| {
                union_counts_with_compressed(&acc, bytes, total, &mut scratch)
            })?;
        } else {
            with_count_bytes(&counts_obj, |bytes| {
                merge_counts_with_compressed(&acc, bytes, true, &mut scratch)
            })?;
        }
        std::mem::swap(&mut acc, &mut scratch);
    }

    rle_to_object(py, &Rle { h, w, counts: acc })
}

#[pyfunction]
pub fn area<'py>(
    py: Python<'py>,
    rle_objs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let list = rle_objs.downcast::<PyList>()?;
    let mut values = Vec::with_capacity(list.len());
    for item in list.iter() {
        let d = item.downcast::<PyDict>()?;
        // Preserve the existing validation that every RLE has a two-element
        // size, even though area itself does not use the dimensions.
        let _ = rle_dimensions(d)?;
        let counts_obj = d
            .get_item("counts")?
            .ok_or_else(|| value_error("RLE missing counts"))?;
        values.push(with_count_bytes(&counts_obj, compressed_area)?);
    }
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
    let values = with_count_bytes(counts, bytes_to_counts)?;
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
