//! Adapt raw JPEG component planes to the PDF sample grid.
//!
//! The codec's four-component packed decoder uses padded chroma extents in
//! its upsampler. PDF images need the active extents, including partial MCUs
//! and independent sampling factors. Reuse the codec's public reconstruction
//! and color conversion primitives; never interpolate across MCU padding.
use alloc::{vec, vec::Vec};
use libjpeg_turbo_rs::decode::{
    color::ycck_to_cmyk_row,
    upsample::{fancy_h2v1, fancy_h2v2_row},
};
use libjpeg_turbo_rs::{Decoder, FrameHeader};

struct Plane<'a> {
    samples: &'a [u8],
    stride: usize,
    width: usize,
    height: usize,
    horizontal: usize,
    vertical: usize,
    max_horizontal: usize,
    max_vertical: usize,
}

impl Plane<'_> {
    fn row(&self, y: usize) -> Option<&[u8]> {
        let start = y.checked_mul(self.stride)?;
        self.samples.get(start..start.checked_add(self.width)?)
    }

    fn reconstruct(&self, y: usize, width: usize, out: &mut [u8]) -> Option<()> {
        let sy = y.checked_mul(self.vertical)? / self.max_vertical;
        let current = self.row(sy)?;
        let h2 = self.horizontal.checked_mul(2)? == self.max_horizontal;
        let v2 = self.vertical.checked_mul(2)? == self.max_vertical;
        let full_h = self.horizontal == self.max_horizontal;
        let full_v = self.vertical == self.max_vertical;
        // libjpeg's fancy horizontal kernels are selected only when the
        // *active* input width exceeds two samples. In the 2x2 case this
        // also controls the vertical filter, not merely the horizontal one.
        if h2 && self.width > 2 && full_v {
            fancy_h2v1(current, self.width, out, width);
        } else if h2 && self.width > 2 && v2 {
            let adjacent = if y.is_multiple_of(2) {
                sy.saturating_sub(1)
            } else {
                (sy + 1).min(self.height - 1)
            };
            fancy_h2v2_row(current, self.row(adjacent)?, out, self.width);
        } else if full_h && v2 {
            let even = y.is_multiple_of(2);
            let adjacent = if even {
                sy.saturating_sub(1)
            } else {
                (sy + 1).min(self.height - 1)
            };
            let bias = if even { 1 } else { 2 };
            for ((dst, &a), &b) in out[..width]
                .iter_mut()
                .zip(current)
                .zip(self.row(adjacent)?)
            {
                *dst = ((3 * u16::from(a) + u16::from(b) + bias) >> 2) as u8;
            }
        } else if full_h {
            out.get_mut(..width)?.copy_from_slice(current);
        } else {
            // Integral non-fancy ratios use box replication. For fractional
            // ratios (which libjpeg rejects), retain the PDF multiband
            // recovery path with the same rational sample-grid mapping.
            for (x, dst) in out.get_mut(..width)?.iter_mut().enumerate() {
                *dst = *current.get(x.checked_mul(self.horizontal)? / self.max_horizontal)?;
            }
        }
        Some(())
    }
}

pub(super) fn raw_channels(
    decoder: Decoder<'_>,
    frame: &FrameHeader,
    ycck: bool,
) -> Option<Vec<u8>> {
    let (width, height) = frame.dimensions();
    let count = frame.components.len();
    let max_h = usize::from(
        frame
            .components
            .iter()
            .map(|c| c.horizontal_sampling)
            .max()?,
    );
    let max_v = usize::from(frame.components.iter().map(|c| c.vertical_sampling).max()?);
    if max_h == 0 || max_v == 0 || width == 0 || height == 0 || (ycck && count != 4) {
        return None;
    }
    let raw = decoder.decode_raw().ok()?;
    if raw.width != width || raw.height != height || raw.num_components != count {
        return None;
    }
    let planes: Vec<_> = frame
        .components
        .iter()
        .enumerate()
        .map(|(i, component)| {
            let horizontal = usize::from(component.horizontal_sampling);
            let vertical = usize::from(component.vertical_sampling);
            let active_width = width.checked_mul(horizontal)?.div_ceil(max_h);
            let active_height = height.checked_mul(vertical)?.div_ceil(max_v);
            let stride = *raw.plane_widths.get(i)?;
            let rows = *raw.plane_heights.get(i)?;
            let samples = raw.planes.get(i)?.as_slice();
            if active_width == 0
                || active_height == 0
                || active_width > stride
                || active_height > rows
                || samples.len() < stride.checked_mul(rows)?
            {
                return None;
            }
            Some(Plane {
                samples,
                stride,
                width: active_width,
                height: active_height,
                horizontal,
                vertical,
                max_horizontal: max_h,
                max_vertical: max_v,
            })
        })
        .collect::<Option<_>>()?;
    let stride = width.checked_mul(count)?;
    let length = stride.checked_mul(height)?;
    let mut output = Vec::new();
    output.try_reserve_exact(length).ok()?;
    output.resize(length, 0);
    // Fancy 2x expansion of an odd-width image produces one padding sample.
    let mut rows = vec![vec![0; width.checked_add(1)?]; count];
    for (y, dest) in output.chunks_exact_mut(stride).enumerate() {
        for (plane, row) in planes.iter().zip(&mut rows) {
            plane.reconstruct(y, width, row)?;
        }
        if ycck {
            let [luma, cb, cr, k] = rows.as_slice() else {
                return None;
            };
            ycck_to_cmyk_row(luma, cb, cr, k, dest, width);
        } else {
            for (x, pixel) in dest.chunks_exact_mut(count).enumerate() {
                for (value, row) in pixel.iter_mut().zip(&rows) {
                    *value = row[x];
                }
            }
        }
    }
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_is_not_part_of_the_active_chroma_plane() {
        let samples = [10, 50, 90, 255, 30, 70, 110, 255, 255, 255, 255, 255];
        let plane = Plane {
            samples: &samples,
            stride: 4,
            width: 3,
            height: 2,
            horizontal: 1,
            vertical: 1,
            max_horizontal: 2,
            max_vertical: 2,
        };
        let mut row = [0; 6];
        plane.reconstruct(0, 5, &mut row).unwrap();
        assert_eq!(row, [10, 20, 40, 60, 80, 90]);
        plane.reconstruct(2, 5, &mut row).unwrap();
        assert_eq!(row, [25, 35, 55, 75, 95, 105]);
    }

    #[test]
    fn narrow_chroma_disables_both_fancy_axes() {
        let plane = Plane {
            samples: &[10, 50, 30, 70],
            stride: 2,
            width: 2,
            height: 2,
            horizontal: 1,
            vertical: 1,
            max_horizontal: 2,
            max_vertical: 2,
        };
        let mut row = [0; 4];
        plane.reconstruct(1, 3, &mut row).unwrap();
        assert_eq!(&row[..3], &[10, 10, 50]);
        plane.reconstruct(2, 3, &mut row).unwrap();
        assert_eq!(&row[..3], &[30, 30, 70]);
    }

    #[test]
    fn vertical_only_and_fractional_sampling() {
        let samples = [10, 50, 30, 70];
        let mut plane = Plane {
            samples: &samples,
            stride: 2,
            width: 2,
            height: 2,
            horizontal: 1,
            vertical: 1,
            max_horizontal: 1,
            max_vertical: 2,
        };
        let mut row = [0; 4];
        plane.reconstruct(1, 2, &mut row).unwrap();
        assert_eq!(&row[..2], &[15, 55]);
        plane.reconstruct(2, 2, &mut row).unwrap();
        assert_eq!(&row[..2], &[25, 65]);
        plane.horizontal = 2;
        plane.max_horizontal = 3;
        plane.max_vertical = 1;
        plane.reconstruct(0, 3, &mut row).unwrap();
        assert_eq!(&row[..3], &[10, 10, 50]);
    }
}
