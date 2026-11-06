# Getting the rest of the Unsafe-Net dataset

Status 2026-11-04: **class-0 training clips are downloading. Class 4 is blocked on one
piece of information that must come from a browser.**

## What works

The Mendeley public API serves file metadata and a full listing of any folder whose id is
known:

    https://data.mendeley.com/public-api/datasets/xjmtb22pff/files/<file_id>?version=1
    https://data.mendeley.com/public-api/datasets/xjmtb22pff/files?folder_id=<folder_id>&version=1

The first returns a file's `folder_id`, its sha256 and its download URL. The second returns
every file in that folder with the same details. `scripts/fetch_class_folder.py` takes
either a folder id or any file id inside the wanted folder, lists the folder, downloads
every file and verifies each sha256 against the published one.

The class-0 training folder was discovered this way, from the `0_tr143.mp4` file id already
recorded in `configs/clip_0_tr143.json`:

| Folder | id | Contents |
|---|---|---|
| class 0, train | `7833ca30-4398-4d13-b1c1-b572b8fd6699` | 178 clips, 2.80 GB, all `0_tr*.mp4`, 0 duplicate hashes |

Note that this folder holds the training split only. The dataset is organised per class and
per split, so class 0 test, class 4 train and class 4 test are three further folders.

## What does not work

The dataset root cannot be listed. Every variant tried returns an empty array, a 400 or a
404:

    files?folder_id=root&version=1                      -> []
    files?folder_id=root&version=1&include_folders=true -> []
    files?version=1                                     -> {"error":400}
    folders?version=1                                   -> Cannot GET
    archive?version=1                                   -> Cannot GET

There is no documented endpoint that enumerates a dataset's folders, and folder ids are
random UUIDs, so they cannot be derived or guessed. The dataset landing page is rendered by
JavaScript and its HTML contains no file list.

So a folder can be listed once something inside it is known, but the first thing inside it
has to come from somewhere else.

## What is needed, and it is small

Open <https://data.mendeley.com/datasets/xjmtb22pff/1> in a browser and, for each of the
three remaining folders, copy **any one file's download link**. Right-click a single `.mp4`
and copy the link address; it looks like

    https://data.mendeley.com/public-files/datasets/xjmtb22pff/files/f48e2cbc-bd0e-42b4-bb7e-9ca3980143be/file_downloaded

The UUID in the middle is all that is needed. One link per folder:

- one `4_tr*.mp4` (class 4, train)
- one `4_te*.mp4` (class 4, test)
- one `0_te*.mp4` (class 0, test)

Alternatively, navigating into a folder in the browser puts a `folder_id` in the address
bar; those three folder ids do just as well.

Then, for each:

    python scripts/fetch_class_folder.py --from-file-id <uuid> --out data/raw/class4_train
    python scripts/fetch_class_folder.py --from-file-id <uuid> --out data/raw/class4_test
    python scripts/fetch_class_folder.py --from-file-id <uuid> --out data/raw/class0_test

Each run lists its folder, downloads every file, verifies every sha256 and writes
`folder_manifest.json` beside the clips.

## Why the whole dataset is not downloaded instead

Only classes 0 and 4 are used. The classifier is binary, *Safe Walkway Violation* against
*Safe Walkway*, so the other six classes would add 7 GB of clips that no stage reads.
`CLAUDE.md` also asks that full datasets not be pulled down without need.

## Licence and handling

CC BY 4.0, doi:10.17632/xjmtb22pff.1, Önal & Dandıl, Bilecik Şeyh Edebali University.
Training on it is permitted with attribution. The clips show identifiable workers who
consented to the authors' study and not to third-party redistribution, so `data/` stays
gitignored and nothing is republished.

---

## Bulk "Download All" route — tried 2026-11-04, does not work

The landing page's Download All button is driven by an endpoint that needs the dataset's
**root folder id**, and that id is exactly what cannot be obtained. Every route tried:

| Route | Result |
|---|---|
| `files/archive?version=1` | `{"error":404}` |
| `files/archive?version=1&folder_id=root` | `{"error":404}` |
| `public-files/datasets/xjmtb22pff/files/archive/file_downloaded` | `{"code":400,"message":"Bad Request"}` |
| `public-api/archives/datasets/xjmtb22pff?version=1` | `{"error":404}` |
| `public-api/datasets/xjmtb22pff/archives?version=1` | `{"error":404}` |

The literal string `root` is accepted by the file-listing route, where it returns an empty
array, and rejected by the archive route with a 404. So there is no spelling of the root
folder that both routes accept, and the archive cannot be requested for the whole dataset.

Pagination was ruled out as an explanation for the class-0 folder holding 178 files rather
than the paper's 566 training clips. The listing returns the same 178 entries under
`limit=1000`, under `limit=1000&offset=178` and under `page=2`, first entry `0_tr100.mp4`
and last `0_tr9.mp4` in every case. The folder genuinely holds one class and one split.

One incidental finding, reported rather than used: the `files/archive/file_downloaded`
error body includes a service bearer token belonging to Mendeley's own client. It has not
been used, stored or recorded here, and no request in this project carries it.

## Browser download, and transferring the clips to Bridges-2

If supplying the three UUIDs is inconvenient, download in a browser and transfer instead.

1. Open <https://data.mendeley.com/datasets/xjmtb22pff/1> and use **Download All** (10.0 GB),
   or open each class folder and download the three needed ones.
2. Unzip locally. Keep the folder names and the original filenames; the `<classID>_<tr|te><index>.mp4`
   convention is what every script here reads the label from.
3. Transfer only the classes this experiment uses. From the Mac:

        rsync -avP <local>/0_te <local>/4_tr <local>/4_te \
          <user>@data.bridges2.psc.edu:/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/data/raw/

   Use `data.bridges2.psc.edu`, not the login node. Expect a password and 2FA prompt.

4. Afterwards the clips carry no checksum record, because the published sha256 values live
   behind the same API listing that cannot be reached without the folder id. Provenance is
   then "downloaded via browser on <date>" rather than "verified against the publisher's
   sha256". That is a real, if small, weakening of the record and it is noted in the run
   manifests rather than glossed over.

Supplying three UUIDs keeps the checksum verification. Transferring a browser download does
not. Either is workable; they are not equivalent.

---

## Update 2026-11-05: classes 2 and 6 are blocked the same way

The panel-cover experiment needs the class-2 and class-6 **training** folders. Classes 0 and
4 are already on disk; the official test split is deliberately left untouched.

Expected volume, from Table 2 of the paper: class 2 has 129 training clips averaging 5.8 s,
class 6 has 19 training clips averaging 4.6 s. Scaling from class 0, which is 178 clips at
9.7 s for 2.80 GB, these two folders should be roughly 1.3 GB and 0.2 GB.

### Everything tried today, and what it returned

The root is still unlistable, and three further routes were tried and failed:

| Attempt | Result |
|---|---|
| `files?version=1&include_folders=true` | `{"error":400}` |
| `GET /public-api/datasets/xjmtb22pff?version=1` | 200, full dataset metadata, **no file or folder list**, `size: 0` |
| `/public-api/datasets/xjmtb22pff/versions/1/files` | `Cannot GET` |
| `api.data.mendeley.com/datasets/xjmtb22pff/versions/1/files` | `no Route matched with those values` |
| `/public-api/datasets/xjmtb22pff/folders?version=1&folder_id=root` | `Cannot GET` |
| Landing-page HTML fetched with a browser user agent | 128,943 bytes, **zero occurrences of `mp4`**, no `__NEXT_DATA__` payload |
| Every UUID appearing in the metadata or page HTML, tried as both `folder_id` and file id | `{"error":404}` for all five; they are licence, institution and category ids |
| `prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/xjmtb22pff-1.zip` | `403 Forbidden` from S3, so no ranged read of a cached archive either |

The S3 route is worth noting because it would have solved this without any browser step: a
cached zip that honoured HTTP range requests would let the central directory be read and only
the wanted members fetched. That bucket does not serve this dataset publicly.

### What is needed: two links

Open <https://data.mendeley.com/datasets/xjmtb22pff/1>, enter the class-2 training folder and
the class-6 training folder, and copy **any one file's download link from each**:

- one `2_tr*.mp4` link, for Opened Panel Cover, training
- one `6_tr*.mp4` link, for Closed Panel Cover, training

A link looks like

    https://data.mendeley.com/public-files/datasets/xjmtb22pff/files/<uuid>/file_downloaded

The UUID is the only part needed. A `folder_id` copied from the address bar while inside the
folder works just as well. Then:

    python scripts/fetch_class_folder.py --from-file-id <uuid-of-a-2_tr-clip> --out data/raw/class2_train
    python scripts/fetch_class_folder.py --from-file-id <uuid-of-a-6_tr-clip> --out data/raw/class6_train

Each run lists the folder, downloads every file, verifies every published sha256 and writes
`folder_manifest.json` beside the clips. Nothing about this step touches a GPU, so it runs on
the login node without charge.
