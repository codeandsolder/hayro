use crate::WORKSPACE_PATH;
use crate::harness::args::args;
use crate::harness::assets;
use crate::harness::snapshot::encode_png;
use image::{RgbaImage, load_from_memory};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex, PoisonError};

static LOCKFILE: LazyLock<BTreeMap<String, String>> = LazyLock::new(|| {
    let Ok(content) = std::fs::read_to_string(lockfile_path()) else {
        return BTreeMap::new();
    };

    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let (key, hash) = line
                .split_once(' ')
                .unwrap_or_else(|| panic!("invalid line in references.lock: {line}"));
            (key.to_string(), hash.to_string())
        })
        .collect()
});

static RENDERED: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

static DOWNLOAD: Mutex<()> = Mutex::new(());

pub(crate) fn lockfile_path() -> PathBuf {
    WORKSPACE_PATH.join("references.lock")
}

fn cache_path(hash: &str) -> PathBuf {
    WORKSPACE_PATH
        .join("references")
        .join(format!("{hash}.png"))
}

/// A lockfile entry identifies both immutable pixels and their publishing origin.
#[derive(Clone, Copy)]
pub(crate) struct Reference<'a> {
    pub(crate) hash: &'a str,
    entry: &'a str,
    pub(crate) origin: Origin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Origin {
    Upstream,
    Fork,
}

impl<'a> Reference<'a> {
    fn parse(entry: &'a str) -> Option<Self> {
        let (hash, origin) = entry
            .strip_prefix("fork:")
            .map_or((entry, Origin::Upstream), |hash| (hash, Origin::Fork));
        if hash.len() != 32
            || !hash
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return None;
        }
        Some(Self {
            hash,
            entry,
            origin,
        })
    }
}

pub(crate) fn expected(key: &str) -> Option<Reference<'static>> {
    LOCKFILE.get(key).map(|entry| {
        Reference::parse(entry)
            .unwrap_or_else(|| panic!("invalid reference identity for {key}: {entry}"))
    })
}

pub(crate) fn hash(image: &RgbaImage) -> String {
    let mut hasher = Sha256::new();
    hasher.update(image.width().to_le_bytes());
    hasher.update(image.height().to_le_bytes());
    hasher.update(image.as_raw());

    hasher.finalize()[..16]
        .iter()
        .fold(String::new(), |mut out, byte| {
            write!(out, "{byte:02x}").unwrap();
            out
        })
}

pub(crate) fn record(key: &str, hash: &str, image: &RgbaImage) {
    let Some(dir) = &args().ci else { return };

    let unchanged = expected(key).filter(|reference| reference.hash == hash);
    // Regeneration must not lose the source of an unchanged fork reference.
    let entry = unchanged.map_or(hash, |reference| reference.entry);
    RENDERED
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(key.to_string(), entry.to_string());

    if unchanged.is_none() {
        let path = dir.join("images").join(format!("{hash}.png"));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, encode_png(image)).unwrap();
    }
}

pub(crate) fn load(reference: Reference<'_>) -> Result<(RgbaImage, PathBuf), String> {
    let path = cache_path(reference.hash);

    if !path.exists() {
        if args().offline {
            return Err("isn't downloaded (offline)".to_string());
        }

        let _guard = DOWNLOAD.lock().unwrap_or_else(PoisonError::into_inner);
        if !path.exists() {
            let relative = path.strip_prefix(&*WORKSPACE_PATH).unwrap();
            assets::download_reference(reference, relative)
                .map_err(|err| format!("couldn't be downloaded: {err}"))?;
        }
    }

    let data = std::fs::read(&path).map_err(|err| format!("couldn't be read: {err}"))?;
    let image = load_from_memory(&data)
        .map_err(|err| format!("is invalid: {err}"))?
        .into_rgba8();

    let actual = hash(&image);
    if actual != reference.hash {
        return Err(format!(
            "has incorrect content hash {actual}, expected {}",
            reference.hash
        ));
    }

    Ok((image, path))
}

pub(crate) fn write_lockfile(dir: &Path) {
    let rendered = RENDERED.lock().unwrap_or_else(PoisonError::into_inner);
    let content = rendered.iter().fold(String::new(), |mut out, (key, hash)| {
        writeln!(out, "{key} {hash}").unwrap();
        out
    });
    std::fs::write(dir.join("references.lock"), content).unwrap();
}

/// Exercise source selection and lockfile grammar independently of network availability.
pub(crate) fn test_reference_identity() {
    let hash = "0123456789abcdef0123456789abcdef";
    let fork = format!("fork:{hash}");
    let upstream = Reference::parse(hash).unwrap();
    let fork = Reference::parse(&fork).unwrap();
    assert_eq!(upstream.hash, fork.hash);
    assert_eq!(upstream.origin, Origin::Upstream);
    assert_eq!(fork.origin, Origin::Fork);
    assert_eq!(fork.entry, format!("fork:{hash}"));
    for invalid in [
        "",
        "fork:",
        "unknown:0123456789abcdef0123456789abcdef",
        "../../a.png",
        "0123456789abcdef0123456789abcdeF",
    ] {
        assert!(Reference::parse(invalid).is_none());
    }
}
