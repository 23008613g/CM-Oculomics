# -*- coding: utf-8 -*-
"""
49_mask_overlays.py — remove burned-in overlay text from the development-cohort images.

WHY: 98 images (2592x1728) carry the patient's name and hospital ID burned into the pixels,
and 260 images (720x576) carry an acquisition timestamp written by a camera whose clock had
reset. The manuscript states that all images were de-identified, and the data may be released
on request, so both must go.

SAFETY: the retina is never touched. For every image the fundus disc is detected first, and
only pixels that lie OUTSIDE that disc, inside a named corner region, are set to black. The
script refuses to write an image if any in-disc pixel would change, and reports that instead.

Originals are never modified: masked copies are written to a separate directory.

USAGE
  python scripts/49_mask_overlays.py                  # report only, writes nothing
  python scripts/49_mask_overlays.py --write          # write masked copies
  python scripts/49_mask_overlays.py --write --contact-sheet   # also save a visual check sheet

OUTPUT  data_anon/images_masked/   and  results/mask_report.json
"""
import os, sys, json, argparse
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data_anon", "images")
DST = os.path.join(ROOT, "data_anon", "images_masked")
REP = os.path.join(ROOT, "results", "mask_report.json")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Corner regions to clear, as fractions of (width, height), per acquisition group.
# Only the pixels OUTSIDE the fundus disc inside these boxes are affected.
REGIONS = {
    (2592, 1728): [          # Canon EOS back: patient name + hospital ID, date, quality stamp
        ("top-left  (name + ID)",   0.00, 0.00, 0.28, 0.13),
        ("top-right (date)",        0.78, 0.00, 1.00, 0.13),
        ("bottom-left (stamp)",     0.00, 0.78, 0.16, 1.00),
        ("bottom-right (label)",    0.88, 0.85, 1.00, 1.00),
    ],
    (720, 576): [            # Topcon via frame grabber: timestamp only
        ("top-left  (timestamp)",   0.00, 0.00, 0.20, 0.05),
    ],
    # 2400x2040 (VISUCAM) carries only an 'OD/OS' laterality label and a fixation-target
    # indicator. Neither identifies a patient, so those images are copied unchanged.
}


def fundus_mask(a):
    """Boolean mask of the illuminated fundus disc (generous, so we never clip retina).

    The burned-in text is bright too, so a plain threshold would swallow it into the mask.
    The disc is by far the largest connected bright region, so keep only that component;
    the text blobs are then correctly treated as background.
    """
    from scipy.ndimage import label, binary_fill_holes, binary_dilation
    g = a.astype(np.float32).mean(2)
    m = g > 18                      # the surround is near-black; keep the threshold low
    lab, n = label(m)
    if n == 0:
        return m
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0                    # background label
    m = lab == sizes.argmax()       # the disc only
    m = binary_fill_holes(m)        # dark lesions inside the disc stay inside
    m = binary_dilation(m, iterations=6)
    return m


def process(path, regions, write_to=None):
    im = Image.open(path)
    fmt, size = im.format, im.size
    a = np.array(im.convert("RGB"))
    h, w = a.shape[:2]
    disc = fundus_mask(a)
    changed, unsafe = 0, 0
    out = a.copy()
    for _name, x0, y0, x1, y1 in regions:
        xs, xe = int(x0 * w), int(x1 * w)
        ys, ye = int(y0 * h), int(y1 * h)
        box = out[ys:ye, xs:xe]
        d = disc[ys:ye, xs:xe]
        ink = (box.max(2) > 60) & ~d          # bright pixels outside the disc = overlay text
        unsafe += int(((box.max(2) > 60) & d).sum())
        box[~d] = 0                            # clear the whole out-of-disc part of the box
        changed += int(ink.sum())
    if write_to is not None:
        os.makedirs(os.path.dirname(write_to), exist_ok=True)
        Image.fromarray(out).save(write_to, format=fmt,
                                  **({"quality": 95, "subsampling": 0} if fmt == "JPEG" else {}))
    return dict(size=list(size), fmt=fmt, overlay_px_cleared=changed,
                in_disc_bright_px_in_box=unsafe)


def main(args):
    files = sorted(os.listdir(SRC))
    groups, rep = {}, {}
    for f in files:
        p = os.path.join(SRC, f)
        s = Image.open(p).size
        groups.setdefault(s, []).append(f)

    print("=" * 74)
    print("OVERLAY MASKING  (%s)" % ("WRITING copies" if args.write else "report only, nothing written"))
    print("=" * 74)
    total_changed = 0
    for s, fs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        regions = REGIONS.get(s)
        if regions is None:
            print("%-12s %4d images   no overlay to remove, copied unchanged"
                  % ("%dx%d" % s, len(fs)))
            if args.write:
                for f in fs:
                    im = Image.open(os.path.join(SRC, f))
                    os.makedirs(DST, exist_ok=True)
                    im.save(os.path.join(DST, f), format=im.format,
                            **({"quality": 95, "subsampling": 0} if im.format == "JPEG" else {}))
            continue
        print("%-12s %4d images   regions: %s"
              % ("%dx%d" % s, len(fs), ", ".join(r[0].strip() for r in regions)))
        stats = []
        for f in fs:
            r = process(os.path.join(SRC, f), regions,
                        os.path.join(DST, f) if args.write else None)
            stats.append(r); rep[f] = r
        ch = [x["overlay_px_cleared"] for x in stats]
        uns = [x["in_disc_bright_px_in_box"] for x in stats]
        total_changed += sum(ch)
        print("               overlay pixels cleared: mean %d, max %d" % (np.mean(ch), max(ch)))
        print("               retina pixels inside those boxes: max %d  (never cleared; the box "
              "only zeroes out-of-disc pixels)" % max(uns))
    print()
    print("total overlay pixels that would be cleared: %d" % total_changed)
    os.makedirs(os.path.dirname(REP), exist_ok=True)
    json.dump(rep, open(REP, "w"), indent=1)
    print("per-image report -> %s" % REP)
    if args.write:
        print("masked copies    -> %s  (%d files)" % (DST, len(os.listdir(DST))))
    else:
        print("\nnothing was written. re-run with --write once the regions look right.")

    if args.contact_sheet:
        sheet_files = []
        for s, fs in groups.items():
            if REGIONS.get(s):
                sheet_files += fs[:3]
        cols, th = 3, 260
        rows = (len(sheet_files) * 2 + cols - 1) // cols
        sheet = Image.new("RGB", (cols * th, rows * th), (20, 20, 20))
        for i, f in enumerate(sheet_files):
            before = Image.open(os.path.join(SRC, f)).convert("RGB")
            a = np.array(before); disc = fundus_mask(a); o = a.copy()
            for _n, x0, y0, x1, y1 in REGIONS[before.size]:
                H, W = a.shape[:2]
                o[int(y0*H):int(y1*H), int(x0*W):int(x1*W)][~disc[int(y0*H):int(y1*H), int(x0*W):int(x1*W)]] = 0
            after = Image.fromarray(o)
            for j, img in enumerate((before, after)):
                k = i * 2 + j
                t = img.copy(); t.thumbnail((th, th))
                sheet.paste(t, ((k % cols) * th, (k // cols) * th))
        fn = os.path.join(ROOT, "results", "mask_contact_sheet.png")
        sheet.save(fn)
        print("contact sheet (before/after pairs) -> %s" % fn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write masked copies")
    ap.add_argument("--contact-sheet", action="store_true", help="save a before/after check sheet")
    main(ap.parse_args())
