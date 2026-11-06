# Verified definitions and provenance: class 2 against class 6

Written 2026-11-05 for the panel-cover counterfactual experiment. Every statement on this
page was read today from the publisher record or the peer-reviewed paper, and the exact
source is named. Nothing here is carried from memory, and nothing is inferred from the
class names alone.

## Sources consulted today

| Source | Retrieved | What it settled |
|---|---|---|
| Mendeley public API, `GET /public-api/datasets/xjmtb22pff?version=1` | 2026-11-05 | Dataset title, description, collection method, licence, contributors, institution, publication date |
| Paper, Table 1, `link.springer.com/article/10.1007/s11042-024-19276-8/tables/1` | 2026-11-05 | The class ID to class name mapping, and the safe/unsafe type of each class |
| Paper, Table 2, same article, `/tables/2` | 2026-11-05 | Per-class training and test counts, and average clip duration |
| Paper body text, same article | 2026-11-05 | The behavioural definition of each class, and the figure captions that show what the panel looks like |

## The two classes, as the authors define them

Table 1 of the paper gives the class ID mapping. The dataset's filename convention
`<classID>_<tr|te><index>.mp4` uses these same IDs.

| Class ID | Class name | Behaviour type |
|---|---|---|
| 2 | Opened Panel Cover | Unsafe |
| 6 | Closed Panel Cover | Safe |

The paper's body text defines the pair in one sentence, quoted exactly:

> "Similarly, leaving panel covers open/forgetting them is classified as Opened Panel Cover,
> an unsafe behaviour, while closed panel covers are classified as Closed Panel Cover, a
> safe behaviour."

The figure captions identify what the object is. Figure 2(d) is described as "a safe
behaviour for the Closed Panel Cover class for a panel connected to a machine", and Figure
2(h) as "an unsafe behaviour for the Opened Panel Cover class".

### What these definitions do and do not license

* The distinction is a **state of an object**, not an action by a person. The unsafe class is
  the cover being left open or forgotten, so a clip can carry the unsafe label with no worker
  visible at all. This is the opposite of the class 0 / class 4 pair used earlier in this
  project, where the label is a property of where a person walks.
* The paper never states that a closed cover makes a scene safe overall. Figure 9(a) is
  described as one video containing frames of the unsafe class Safe Walkway Violation
  together with frames of the safe classes Authorized Intervention, **Closed Panel Cover**
  and Safe Walkway. So a closed panel coexists with an unsafe scene in the authors' own
  example. **Closing a panel in a clip is therefore not evidence that the clip became safe.**
* Equally, the paper does not state that an open cover is unsafe in all contexts. A cover may
  be legitimately open during authorized maintenance; the authors' own class 5, Authorized
  Intervention, exists for permitted work. The label records "left open or forgotten", which
  is a judgement the annotators made and which no measurement in this project can re-derive.
* The paper reports that Opened Panel Cover is one of the two classes with the higher YOLO v4
  false-positive counts, and that Opened Panel Cover and Unauthorized Intervention were
  confused with each other by Unsafe-Net (Figures 12(e) and 12(f)). The authors' own system
  does not treat this class as trivially separable.

## Data volume, from Table 2

| Class | Training clips | Test clips | Average duration |
|---|---|---|---|
| 2, Opened Panel Cover | 129 | 13 | 5.8 s |
| 6, Closed Panel Cover | **19** | 13 | 4.6 s |
| 0, Safe Walkway Violation | 178 | 32 | 9.7 s |
| 4, Safe Walkway | 50 | 25 | 9.2 s |

Table 2's counts for classes 0 and 4 match the folders already downloaded to
`data/raw/class0_train` (178 files) and `data/raw/class4_train` (50 files) exactly, which is
independent confirmation that Table 2 describes the published folders and not some internal
subset.

**Confirmed again on 2026-11-05 for this experiment's two classes.** The class-2 training
folder `22df4661-d1db-40b8-a429-9203bdffdbd2` listed 129 files totalling 1.37 GB, and the
class-6 training folder `dca5cd74-fbb9-4226-aaac-3021d67d8a6f` listed 19 files totalling
0.16 GB. Both were downloaded in full and every file matched the sha256 the Mendeley API
publishes, with no failures. Four of the eight published training folders now match Table 2
exactly.

**The consequence to plan around.** Class 6 has 19 training clips. After recording-group
separation, which is mandatory here because the dataset is two fixed cameras in one factory,
the number of independent class-6 recordings available for training will be smaller still.
This is a small-data problem by construction, and it is known before a single clip is
downloaded rather than discovered after a classifier has been trained.

## Recording conditions, from the publisher record

* Two IP cameras, UNV `IPC2122CR3-PF40-A`, fixed surveillance mounts.
* 1920x1080, H.264, **24 fps** as declared, over roughly 4000 hours across 39 days.
* Collected at Kafaoğlu Metal Plastik Makine San. ve Tic. A.Ş., Eskişehir, Turkey.
* The Mendeley record gives the collection window as 5 November to 13 December 2022. The
  paper says the original dataset was built from fragments collected between 5 November and
  23 November 2022. Both are recorded here; they are not reconciled, and no clip-level date
  is available from either source.
* Permissions were obtained from company officials and employees for the authors' research
  study. That is consent to their study, not to redistribution by third parties, so the
  project's existing decision to keep clips out of version control and out of hosted pages
  stands unchanged.

## Two corrections to this project's existing records

1. **First author's given name.** `README.md` and `data/raw/PROVENANCE.json` give the first
   author as "Onur Önal". Both the Mendeley contributor record and the paper's author list
   give **"Oğuzhan Önal"**. The paper is authoritative; the existing records are wrong on
   this point and the citation should read Önal, O. with "Oğuzhan" spelled out.
2. **Frame rate.** The publisher's method statement declares 24 fps for the dataset, while
   this project's probe of `0_tr143.mp4` recorded 25 fps. A per-file probe beats a
   dataset-level declaration, so the per-clip rate must be read from each file rather than
   assumed. Panel-cover clips average 5.8 s and 4.6 s, so at either rate a 49-frame
   CogVideoX window at stride 3 needs 145 source frames, which is about 5.8 s at 25 fps.
   **Clip length is tight for this class pair and must be checked per clip.**

## Licence and attribution

CC BY 4.0, `http://creativecommons.org/licenses/by/4.0`. Dataset DOI
`10.17632/xjmtb22pff.1`. Paper DOI `10.1007/s11042-024-19276-8`, Multimedia Tools and
Applications (2025) 84:34967-34993. The licence text in the publisher record notes that
further permission may be required for third-party content within the dataset.
