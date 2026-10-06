"""Prepare the six article datasets from the publishers' original files."""
from argparse import ArgumentParser
import csv
import io
from pathlib import Path
import re
import zipfile

import numpy as np
from scipy import ndimage as ndi
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

from references import reference_file

ROOT = Path(__file__).resolve().parent
SUBJECTS = {'rice': 1, 'corn': 1, 'wheat': 1, 'capgmyo': 18, 'hyser': 20, 'flex': 13}
RICE_CLASSES = ('91RH', 'CNC12', 'GS55R', 'HT18', 'LDA8', 'LTH35',
                'N54', 'NM14', 'NepKB19', 'PD211')
RICE_IMAGES = ('LTH35', 'NepKB19', 'NM14-2', '91RH', 'LDA8', 'N54',
               'PD211', 'CNC12', 'HT18', 'GS55R')
FLEX_GESTURES = ('abduct_p1', 'adduct_p1', 'extend_p1', 'grip_p1', 'pronate_p1',
                 'rest_p1', 'supinate_p1', 'tripod_p1', 'wextend_p1', 'wflex_p1')


def metadata(dataset, subject):
    with np.load(reference_file(f'splits/{dataset}_{subject:02d}.npz', ROOT),
                 allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def check_order(meta, labels, counts, groups=None, trial_ids=None):
    """Reject a different release, object order, or segmentation before fitting."""
    np.testing.assert_array_equal(labels, meta['labels'], err_msg='Object labels/order differ')
    np.testing.assert_array_equal(np.asarray(counts)-1, meta['dof'], err_msg='Sample counts differ')
    if groups is not None:
        np.testing.assert_array_equal(groups, meta['groups'], err_msg='Physical grain groups differ')
    if trial_ids is not None and 'trial_ids' in meta:
        np.testing.assert_array_equal(trial_ids, meta['trial_ids'], err_msg='Contraction order differs')


def largest_component(mask):
    labels, count = ndi.label(mask)
    if not count:
        raise ValueError('Empty foreground mask')
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def corn_mask(png):
    gray = png[..., 0] if png.ndim == 3 else png
    corners = np.r_[gray[:2, :2].ravel(), gray[:2, -2:].ravel(),
                    gray[-2:, :2].ravel(), gray[-2:, -2:].ravel()]
    mask = gray > 51 if np.median(corners) <= 51 else gray < 51
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    return ndi.binary_fill_holes(largest_component(mask))


def envi_cube(header):
    text = header.read_text(encoding='latin-1')
    fields = {}
    for name in ('samples', 'lines', 'bands', 'header offset', 'data type', 'interleave', 'byte order'):
        match = re.search(rf'(?im)^\s*{name}\s*=\s*([^\r\n]+)', text)
        if match is None:
            raise ValueError(f'Missing ENVI field: {name}')
        fields[name] = match[1].strip().lower()
    dtype = {1: 'u1', 2: 'i2', 4: 'f4', 5: 'f8', 12: 'u2'}[int(fields['data type'])]
    endian = '<' if fields['byte order'] == '0' else '>'
    raw = np.memmap(header.with_suffix('.raw'), mode='r', dtype=endian+dtype,
                    offset=int(fields['header offset']))
    h, w, p = (int(fields[key]) for key in ('lines', 'samples', 'bands'))
    if raw.size != h*w*p:
        raise ValueError('Unexpected ENVI file size')
    if fields['interleave'] == 'bil':
        return raw.reshape(h, p, w).transpose(0, 2, 1)
    if fields['interleave'] == 'bsq':
        return raw.reshape(p, h, w).transpose(1, 2, 0)
    if fields['interleave'] == 'bip':
        return raw.reshape(h, w, p)
    raise ValueError('Unsupported ENVI interleave')


def rice_boxes(cube):
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import closing
    from skimage.segmentation import clear_border
    band = np.asarray(cube[:, :, -1], dtype=np.float32)
    labels = label(clear_border(closing(band > threshold_otsu(band), np.ones((3, 3)))))
    boxes = []
    for region in regionprops(labels):
        if region.area < 50:
            continue
        r0, c0, r1, c1 = region.bbox
        h, w = r1-r0, c1-c0
        boxes.append((max(0, int(r0-.35*h)), max(0, int(c0-.15*w)),
                      min(cube.shape[0], int(r1+.35*h)), min(cube.shape[1], int(c1+.15*w))))
    return boxes


def rice_mask(crop):
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_closing, binary_erosion, binary_opening, disk, remove_small_objects
    x = np.asarray(crop, dtype=np.float32)
    border = np.zeros(x.shape[:2], bool)
    border[:2] = border[-2:] = True
    border[:, :2] = border[:, -2:] = True
    background_spectra = x[border]
    background = np.median(background_spectra, axis=0)
    scale = 1.4826*np.median(np.abs(background_spectra-background), axis=0)
    positive = scale[scale > 0]
    reference = float(np.median(positive)) if positive.size else 1.
    scale = np.maximum(scale, max(reference*.1, 1.))
    score = np.sqrt(np.mean(((x-background)/scale)**2, axis=2))
    mask = binary_opening(score > threshold_otsu(score), disk(1))
    mask = remove_small_objects(ndi.binary_fill_holes(binary_closing(mask, disk(1))), min_size=20)
    labels = label(mask)
    regions = regionprops(labels)
    if not regions:
        raise ValueError('No RICE grain detected')
    mask = labels == max(regions, key=lambda region: region.area).label
    eroded = binary_erosion(mask, disk(1))
    return eroded if eroded.any() else mask


def rice_objects(source):
    with (source/'index.csv').open(encoding='utf-8-sig', newline='') as stream:
        index = {row['File Name']: row for row in csv.DictReader(stream)}
    for variety in RICE_IMAGES:
        for view in (1, 2):
            name = f'{variety}-{view:02d}'
            row = index[name]
            cube = envi_cube(source/row['Folder']/(name+'.hdr'))
            label = RICE_CLASSES.index('NM14' if variety == 'NM14-2' else variety)
            for box in rice_boxes(cube):
                crop = cube[box[0]:box[2], box[1]:box[3]]
                yield crop[rice_mask(crop)], label, None


def corn_objects(source):
    from PIL import Image
    with zipfile.ZipFile(source/'4n4xbnx8sr-1.zip') as archive:
        names = sorted(n for n in archive.namelist() if n.lower().endswith('.npy'))
        classes = sorted({n.split('/')[2] for n in names})
        for name in names:
            cube = np.load(io.BytesIO(archive.read(name)), allow_pickle=False)
            png = np.asarray(Image.open(io.BytesIO(archive.read(name[:-4]+'.png'))))
            yield cube[corn_mask(png)], classes.index(name.split('/')[2]), None


def wheat_objects(source):
    names = sorted(p for p in source.glob('*/*/*.mat')
                   if p.parent.parent.name in ('fanmai8', 'jinan17', 'xingmai13', 'yangmai6'))
    groups = [f'{p.parent.parent.name}-{p.parent.name.split("-")[1]}-{p.stem}' for p in names]
    unique, counts = np.unique(groups, return_counts=True)
    paired = set(unique[counts == 2])
    classes = sorted({p.parent.parent.name for p in names})
    for path, group in zip(names, groups):
        if group not in paired:
            continue
        cube = loadmat(path)['hyperspectral_data']
        yield cube[np.any(cube != 0, axis=2)], classes.index(path.parent.parent.name), group


def prepare_hsi(dataset, source, output):
    meta = metadata(dataset, 0)
    folder = output/dataset
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'subject00_covariances.npy'
    partial = path.with_suffix('.partial.npy')
    p = 256 if dataset == 'rice' else 300
    covariance = np.lib.format.open_memmap(partial, mode='w+', dtype='float32',
                                          shape=(len(meta['labels']), p, p))
    labels, counts, groups = [], [], []
    iterator = {'rice': rice_objects, 'corn': corn_objects, 'wheat': wheat_objects}[dataset](source)
    for i, (pixels, label, group) in enumerate(iterator):
        n = len(pixels)
        if i >= len(covariance) or n-1 != meta['dof'][i] or label != meta['labels'][i]:
            raise ValueError(f'{dataset}: object {i} differs from the article selection')
        x = np.asarray(pixels, dtype=np.float64)
        x -= x.mean(axis=0, keepdims=True)
        # Preserve the published float32 SCMs; the runner applies n/(n-1) where needed.
        denominator = n if str(meta['source_denominator']) == 'n' else n-1
        covariance[i] = x.T@x/denominator
        labels.append(label)
        counts.append(n)
        groups.append(group)
        if (i+1) % 100 == 0:
            print(f'{dataset}: {i+1}/{len(covariance)} SCMs', flush=True)
    check_order(meta, labels, counts, groups if dataset == 'wheat' else None)
    covariance.flush()
    del covariance
    partial.replace(path)
    np.savez_compressed(folder/'subject00.npz', **meta)
    print(f'{dataset}: prepared {len(labels)} SCMs', flush=True)


def capgmyo_signals(source, subject):
    signals, labels, ids = [], [], []
    with zipfile.ZipFile(source/f'dba-s{subject}.zip') as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith('.mat'):
                continue
            item = loadmat(io.BytesIO(archive.read(name)))
            x = np.asarray(item['data'], dtype=np.float64)
            gesture, trial = int(item['gesture'].item()), int(item['trial'].item())
            if x.shape != (1000, 128) or int(item['subject'].item()) != subject:
                raise ValueError('Unexpected CapgMyo record')
            signals.append(x[436:563:2])
            labels.append(gesture)
            ids.append(f'gesture{gesture:02d}/trial{trial:02d}')
    return np.stack(signals), labels, ids


def hyser_signals(source, subject):
    folder = source/f'subject{subject:02d}_session1'
    labels = np.loadtxt(folder/'label_dynamic.txt', delimiter=',', dtype=int).ravel()
    signals, ids = [], []
    for i in range(1, len(labels)+1):
        name = f'dynamic_preprocess_sample{i}'
        lines = [line for line in (folder/(name+'.hea')).read_text().splitlines()
                 if line and not line.startswith('#')]
        record, channels, frequency, samples = lines[0].split()[:4]
        if (int(channels), float(frequency), int(samples)) != (256, 2048., 2048):
            raise ValueError('Unexpected Hyser record dimensions')
        gains, baselines = [], []
        for line in lines[1:]:
            fields = line.split()
            if fields[0] != record+'.dat' or fields[1] != '16':
                raise ValueError('Expected single-file WFDB format 16')
            match = re.fullmatch(r'([^(/]+)(?:\(([-+\d]+)\))?/([^ ]+)', fields[2])
            if match is None:
                raise ValueError('Invalid WFDB gain/baseline')
            gains.append(float(match[1]))
            baselines.append(int(match[2]) if match[2] is not None else int(fields[4]))
        if len(gains) != 256 or np.any(np.asarray(gains) <= 0):
            raise ValueError('Invalid WFDB channels or gains')
        raw = np.memmap(folder/(name+'.dat'), dtype='<i2', mode='r', shape=(2048, 256))
        digital = np.asarray(raw[1410:1663:4], dtype=np.float64)
        if np.any(digital == -32768):
            raise ValueError('Missing WFDB samples')
        signals.append((digital-np.asarray(baselines))/np.asarray(gains))
        ids.append(name+'.npz')
    return np.stack(signals), labels, ids


def flex_signals(source, subject):
    import h5py
    signals, labels, ids = [], [], []
    with h5py.File(source/f'p{subject:03d}/data_allchannels_initial.h5', 'r') as archive:
        for gesture, name in enumerate(FLEX_GESTURES, 1):
            values = archive[name]
            if values.ndim != 3 or values.shape[1] != 64 or values.shape[2] < 505:
                raise ValueError('Unexpected FlexWear record dimensions')
            start = (values.shape[2]-505)//2
            windows = np.asarray(values[:, :, start:start+505:8], dtype=float).transpose(0, 2, 1)
            signals.extend(windows)
            labels.extend([gesture]*len(windows))
            ids.extend(f'initial/{name}/trial{i+1}' for i in range(len(windows)))
    return np.stack(signals), labels, ids


def prepare_emg(dataset, subject, source, output):
    meta = metadata(dataset, subject)
    signals, labels, ids = {'capgmyo': capgmyo_signals, 'hyser': hyser_signals,
                            'flex': flex_signals}[dataset](source, subject)
    if signals.shape[1] != 64 or not np.isfinite(signals).all():
        raise ValueError('Expected finite 64-observation windows')
    check_order(meta, labels, np.full(len(labels), 64), trial_ids=ids)
    folder = output/dataset
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/f'subject{subject:02d}.npz', signals=signals, **meta)
    print(f'{dataset}, subject {subject}: prepared {len(signals)} windows', flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=SUBJECTS)
    parser.add_argument('--source', type=Path, required=True,
                        help='Directory containing the dataset release; see README')
    parser.add_argument('--output', type=Path, default=ROOT/'data')
    parser.add_argument('--subjects', type=int, nargs='+', help='EMG participants to prepare; default all')
    args = parser.parse_args()
    if args.dataset in ('rice', 'corn', 'wheat'):
        if args.subjects:
            parser.error('--subjects is only used for EMG')
        prepare_hsi(args.dataset, args.source, args.output)
    else:
        selected = args.subjects or range(1, SUBJECTS[args.dataset]+1)
        if not set(selected) <= set(range(1, SUBJECTS[args.dataset]+1)):
            parser.error('Unknown participant')
        for subject in selected:
            prepare_emg(args.dataset, subject, args.source, args.output)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
