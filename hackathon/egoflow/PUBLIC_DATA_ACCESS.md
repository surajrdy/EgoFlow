# Public data-access audit

Checked: 2026-08-15

The 18 selected records are Mecka public-upload episodes. For the hero episode
`69bb1239efeadec2abedad96`:

- `GET /api/egoverse/uploads/<id>/video?redirect=1` returns the public MP4.
- `/api/egoverse/uploads/<id>/zarr` returns 404.
- `/api/egoverse/uploads/<id>/annotations` returns 404.
- `/api/egoverse/uploads/<id>/qwen.annotations` returns 404.
- `/api/egoverse/episodes/<id>` returns `Episode not found` because the record is
  served from the uploads workflow.
- The explorer client bundle references public video endpoints only.
- The `egoverse-data` Modal Volume has no `/datasets` directory; it contains the
  generated EgoFlow caches/manifests/results only.

The official EgoVerse repository documents full dataset downloading through its
cloud-backed SQL/S3 synchronization tooling. Those private catalog/storage
credentials are not available in this workspace. Therefore this run uses public
MP4s and one task-level description only. It does not claim access to dense Zarr or
Qwen annotations, nor hierarchical semantic local progress.

Sources:

- https://partners.mecka.ai/egoverse
- https://github.com/GaTech-RL2/EgoVerse
