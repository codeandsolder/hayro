use crate::Limits;
use crate::object::Dict;
use crate::object::dict::keys::COLOR_TRANSFORM;
use crate::object::stream::{FilterResult, ImageColorSpace, ImageData, ImageDecodeParams};
use alloc::borrow::Cow;
mod components;
use components::raw_channels;
use libjpeg_turbo_rs::{ColorSpace, DecodeLimits, Decoder, PixelFormat};

pub(crate) fn decode(
    data: &[u8],
    params: &Dict<'_>,
    image_params: &ImageDecodeParams,
    limits: Limits,
) -> Option<FilterResult<'static>> {
    if image_params.width > u16::MAX as u32 || image_params.height > u16::MAX as u32 {
        return None;
    }

    // Some JPEGs have bogus oversized dimensions, so patch them from the image dictionary.
    let data = maybe_patch_jpeg_dimensions(data, image_params)?;
    let dimension = limits
        .max_image_dimension
        .bound()
        .unwrap_or(u32::from(u16::MAX));
    let dimension = dimension.min(u32::from(u16::MAX)) as usize;
    let codec_limits = DecodeLimits {
        max_width: dimension,
        max_height: dimension,
        max_pixels: limits.max_image_pixels.bound().unwrap_or(u64::MAX),
        ..DecodeLimits::default()
    };
    let mut decoder = Decoder::new_with_limits(&data, codec_limits).ok()?;
    decoder.set_lenient(true);
    let frame = decoder.header().clone();
    let (width, height) = frame.dimensions();
    let components = frame.components.len();
    let length = width.checked_mul(height)?.checked_mul(components)?;
    if !limits.permits_image(u32::from(frame.width), u32::from(frame.height))
        || !limits.max_decoded_stream_size.permits(length)
    {
        return None;
    }
    if image_params
        .num_components
        .is_some_and(|n| !matches!(n, 1 | 3 | 4) && usize::from(n) != components)
    {
        return None;
    }

    let input = decoder.jpeg_color_space();
    let color_space = match components {
        1 => Some(ImageColorSpace::Gray),
        2 => None,
        3 => Some(ImageColorSpace::Rgb),
        4 => Some(ImageColorSpace::Cmyk),
        _ => return None,
    };
    let decoded = match components {
        2 => raw_channels(decoder, &frame, false)?,
        4 if frame.precision == 8
            && !frame.is_lossless
            && frame.components.windows(2).any(|pair| {
                pair[0].horizontal_sampling != pair[1].horizontal_sampling
                    || pair[0].vertical_sampling != pair[1].vertical_sampling
            }) =>
        {
            raw_channels(decoder, &frame, input == ColorSpace::Ycck)?
        }
        _ => {
            let format = match components {
                1 => PixelFormat::Grayscale,
                3 => PixelFormat::Rgb,
                4 => PixelFormat::Cmyk,
                _ => return None,
            };
            decoder.set_output_format(format);
            if input == ColorSpace::YCbCr
                && params.get::<u8>(COLOR_TRANSFORM).is_some_and(|n| n != 1)
            {
                if frame.precision != 8 || frame.is_lossless {
                    return None;
                }
                decoder.set_output_colorspace(ColorSpace::YCbCr);
            }
            let image = decoder.decode_image().ok()?;
            if image.width != width || image.height != height || image.pixel_format != format {
                return None;
            }
            image.data
        }
    };
    if decoded.len() != length {
        return None;
    }

    Some(FilterResult {
        data: Cow::Owned(decoded),
        image_data: Some(ImageData {
            alpha: None,
            color_space,
            bits_per_component: 8,
            width: u32::from(frame.width),
            height: u32::from(frame.height),
        }),
    })
}

fn maybe_patch_jpeg_dimensions<'a>(
    data: &'a [u8],
    image_params: &ImageDecodeParams,
) -> Option<Cow<'a, [u8]>> {
    let sof_offset = find_sof_marker(data)?;

    let height_offset = sof_offset.checked_add(5)?;
    let width_offset = sof_offset.checked_add(7)?;

    let jpeg_height = u16::from_be_bytes([
        *data.get(height_offset)?,
        *data.get(height_offset.checked_add(1)?)?,
    ]);
    let jpeg_width = u16::from_be_bytes([
        *data.get(width_offset)?,
        *data.get(width_offset.checked_add(1)?)?,
    ]);

    let jpeg_area = (jpeg_width as usize).checked_mul(jpeg_height as usize)?;
    let image_area = (image_params.width as usize).checked_mul(image_params.height as usize)?;
    let need_patch = image_area != 0 && jpeg_area > image_area;

    if !need_patch {
        return Some(Cow::Borrowed(data));
    }

    let target_w = (image_params.width as u16).to_be_bytes();
    let target_h = (image_params.height as u16).to_be_bytes();

    let mut patched = data.to_vec();
    patched[height_offset..height_offset.checked_add(2)?].copy_from_slice(&target_h);
    patched[width_offset..width_offset.checked_add(2)?].copy_from_slice(&target_w);

    Some(Cow::Owned(patched))
}

fn find_sof_marker(data: &[u8]) -> Option<usize> {
    let mut i = 0_usize;

    while i.checked_add(1).is_some_and(|next| next < data.len()) {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }

        let marker = data[i + 1];

        match marker {
            // SOF0–SOF15, excluding DHT, JPG and DAC.
            0xC0..=0xCF if marker != 0xC4 && marker != 0xC8 && marker != 0xCC => {
                return Some(i);
            }
            0xFF => {
                i += 1;

                continue;
            }
            0xD8 | 0xD9 | 0x01 | 0x00 => {
                i += 2;

                continue;
            }
            _ => {
                let len_start = i.checked_add(2)?;
                let len_end = i.checked_add(3)?;
                let seg_len =
                    u16::from_be_bytes([*data.get(len_start)?, *data.get(len_end)?]) as usize;

                i = i.checked_add(2)?.checked_add(seg_len)?;
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::FromBytes;
    use alloc::vec;
    use libjpeg_turbo_rs::{Encoder, Subsampling};

    fn image_params(width: u32, height: u32, components: u8) -> ImageDecodeParams {
        ImageDecodeParams {
            width,
            height,
            num_components: Some(components),
            bpc: Some(8),
            ..ImageDecodeParams::default()
        }
    }

    #[test]
    fn jpeg_rgb_and_raw_ycbcr() {
        let pixels = [203, 61, 17].repeat(16 * 8);
        let jpeg = Encoder::new(&pixels, 16, 8, PixelFormat::Rgb)
            .quality(95)
            .subsampling(Subsampling::S444)
            .encode()
            .unwrap();
        let image_params = image_params(16, 8, 3);
        let rgb = decode(&jpeg, &Dict::default(), &image_params, Limits::DEFAULT).unwrap();
        let raw = decode(
            &jpeg,
            &Dict::from_bytes(b"<< /ColorTransform 0 >>").unwrap(),
            &image_params,
            Limits::DEFAULT,
        )
        .unwrap();
        assert_eq!(rgb.data.len(), pixels.len());
        assert_eq!(raw.data.len(), pixels.len());
        assert!(
            rgb.data
                .as_chunks::<3>()
                .0
                .iter()
                .all(|p| p[0].abs_diff(203) <= 3
                    && p[1].abs_diff(61) <= 3
                    && p[2].abs_diff(17) <= 3)
        );
        assert!(
            raw.data
                .as_chunks::<3>()
                .0
                .iter()
                .all(|p| p[0].abs_diff(98) <= 3
                    && p[1].abs_diff(82) <= 3
                    && p[2].abs_diff(203) <= 3)
        );
        assert!(matches!(
            rgb.image_data.unwrap().color_space,
            Some(ImageColorSpace::Rgb)
        ));
    }

    #[test]
    fn jpeg_gray_and_image_limits() {
        let pixels = vec![111; 16 * 8];
        let jpeg = Encoder::new(&pixels, 16, 8, PixelFormat::Grayscale)
            .quality(100)
            .encode()
            .unwrap();
        let image_params = image_params(16, 8, 1);
        let exact = Limits {
            max_decoded_stream_size: crate::Limit::AtMost(128),
            ..Limits::DEFAULT
        };
        let image = decode(&jpeg, &Dict::default(), &image_params, exact).unwrap();
        assert_eq!(image.data.as_ref(), pixels);
        assert!(matches!(
            image.image_data.unwrap().color_space,
            Some(ImageColorSpace::Gray)
        ));
        for limits in [
            Limits {
                max_decoded_stream_size: crate::Limit::AtMost(127),
                ..exact
            },
            Limits {
                max_image_pixels: crate::Limit::AtMost(127),
                ..exact
            },
            Limits {
                max_image_dimension: crate::Limit::AtMost(15),
                ..exact
            },
        ] {
            assert!(decode(&jpeg, &Dict::default(), &image_params, limits).is_none());
        }
        let inferred = decode(
            &jpeg,
            &Dict::default(),
            &ImageDecodeParams::default(),
            exact,
        )
        .unwrap();
        assert_eq!(inferred.data.as_ref(), pixels);
        assert!(decode(&jpeg[..10], &Dict::default(), &image_params, exact).is_none());
    }

    #[test]
    fn jpeg_cmyk_and_ycck_preserve_ink_samples() {
        let pixels = [20, 91, 170, 212].repeat(16 * 8);
        for source in [ColorSpace::Cmyk, ColorSpace::Ycck] {
            let jpeg = Encoder::new(&pixels, 16, 8, PixelFormat::Cmyk)
                .colorspace(source)
                .quality(100)
                .subsampling(Subsampling::S444)
                .encode()
                .unwrap();
            let image = decode(
                &jpeg,
                &Dict::default(),
                &image_params(16, 8, 4),
                Limits::DEFAULT,
            )
            .unwrap();
            assert!(matches!(
                image.image_data.unwrap().color_space,
                Some(ImageColorSpace::Cmyk)
            ));
            assert_eq!(image.data.len(), pixels.len());
            assert!(
                image
                    .data
                    .iter()
                    .zip(&pixels)
                    .all(|(a, b)| a.abs_diff(*b) <= 3),
                "{source:?}: {:?}",
                &image.data[..4]
            );
        }
    }

    #[test]
    fn jpeg_recovers_missing_eoi() {
        let pixels = [203, 61, 17].repeat(16 * 8);
        let jpeg = Encoder::new(&pixels, 16, 8, PixelFormat::Rgb)
            .quality(95)
            .subsampling(Subsampling::S444)
            .encode()
            .unwrap();
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xff, 0xd9]);
        let params = image_params(16, 8, 3);
        let full = decode(&jpeg, &Dict::default(), &params, Limits::DEFAULT).unwrap();
        let truncated = decode(
            &jpeg[..jpeg.len() - 2],
            &Dict::default(),
            &params,
            Limits::DEFAULT,
        )
        .unwrap();
        assert_eq!(truncated.data, full.data);
    }

    #[test]
    fn jpeg_two_band_retains_both_channels() {
        let mut jpeg = vec![0xff, 0xd8, 0xff, 0xdb, 0, 67, 0];
        jpeg.extend([1; 64]);
        jpeg.extend([0xff, 0xc0, 0, 14, 8, 0, 8, 0, 8, 2, 1, 0x11, 0, 2, 0x11, 0]);
        for table in [0x00, 0x10] {
            jpeg.extend([0xff, 0xc4, 0, 20, table, 1]);
            jpeg.extend([0; 15]);
            jpeg.push(0);
        }
        jpeg.extend([0xff, 0xda, 0, 10, 2, 1, 0, 2, 0, 0, 63, 0, 0x0f, 0xff, 0xd9]);
        let result = decode(
            &jpeg,
            &Dict::default(),
            &image_params(8, 8, 2),
            Limits::DEFAULT,
        )
        .unwrap();
        assert_eq!(result.data.as_ref(), &[128; 128]);
        assert!(result.image_data.unwrap().color_space.is_none());
    }
}
