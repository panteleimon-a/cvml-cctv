# cvml-cctv

## Problem positioning (5 Oct 2026)

### Πρόβλημα (1-line)
Εντοπισμός ανωμαλιών συνέπειας σε CCTV όταν η κάμερα αντιμετωπίζεται ως **time-aware sensor**, με κοινή αξιοποίηση visual temporal cues, timestamps και codec behaviour.

### Σημαντικό (1-line)
Η CCTV έχει σταθερή οπτική· τα timestamps και ο καιρός δίνουν ελλιπείς εξόδους ως prior, ενώ το resampling παραμένει high frequency.

## Γιατί οι έμμεσες λύσεις δεν φτάνουν

- **GT-Loc (ICCV 2025)**: εκτιμά `RGB -> (time, location)` (month–hour), αλλά δεν ελέγχει συνέπεια εικόνας/timestamp/codec και δεν αξιοποιεί bitstream.
- **VideoFACT (WACV 2024)**: frame-level tampering + spatial mask από `RGB x T`, χωρίς timestamp/location και χωρίς consistency έλεγχο metadata.
- **Bitstream CCTV (CVCS 2022)**: ταξινόμηση αλλοίωσης από `GOP x 27` σε compressed domain, χωρίς RGB και χωρίς timestamp consistency.
- **Padilha et al. (IEEE TIFS 2021)**: `(RGB, time, location) -> {consistent, inconsistent}` σε still εικόνα, χωρίς codec behaviour και χωρίς fixed-view motion graph.

## Συγκρίσιμες γραμμές με κοινό πλαίσιο

| Venue | Problem | Target | Dimension | Method |
|---|---|---|---|---|
| ICCV 2025 | when/where captured | hour, month, location | `RGB x 1` + cyclic month–hour | GT-Loc |
| WACV 2024 | falsified frame + where | frame `{0,1}` + `128x128` mask | `RGB x T`, no timestamp | VideoFACT |
| CVCS 2022 | insertion/deletion/duplication/permutation | forged class | `GOP x 27`, not RGB | Bitstream CCTV |
| IEEE TIFS 2021 | image-time-location consistency | consistent vs manipulated | `RGB x 1` + time + location | Padilha et al. |

## Proposed direction

Ενοποιημένο, camera-aware, multimodal framework:

`(RGB x T) + timestamp features + bitstream/GOP features + camera context -> consistency/inconsistency, tampering type, modality attribution`

Βασικές αρχές:

- fixed-view CCTV motion graph (όχι generic video graph),
- ανίχνευση αυξημένου motion ως απόκλιση από normal camera-specific behaviour,
- attention πάνω σε comparison function που αποδίδει motion vectors,
- timestamp + weather fusion για ελλιπή metadata,
- high-frequency handling για resampling artefacts,
- random reimplementation + external validation + code-level proof.

## Novelty

Unified, camera-aware multimodal consistency framework that treats CCTV as a time-aware sensor and jointly models visual temporal cues, timestamps, and codec behaviour to locate consistency anomalies.
