.PHONY: setup auth secrets egoflow-secret deploy serve smoke test egoflow-smoke egoflow-full egoflow-test

setup:
	uv sync

auth:
	uv run modal setup

secrets:
	@echo "No secrets required: EgoFlow uses the public Mecka episode URLs."

egoflow-secret:
	@echo "No secrets required: EgoFlow uses the public Mecka episode URLs."

deploy:
	uv run modal deploy hackathon/egoflow/modal_app.py

serve:
	uv run modal serve hackathon/egoflow/modal_app.py

smoke:
	uv run modal run hackathon/egoflow/modal_app.py --action smoke --max-episodes 2 --max-steps 20

test:
	uv run pytest -q

egoflow-test:
	uv run --with 'numpy>=1.26,<3' --with 'zarr>=2.18,<4' --with 'pillow>=10' pytest -q

egoflow-smoke:
	uv run modal run hackathon/egoflow/modal_app.py --action smoke --max-episodes 2 --max-steps 20

egoflow-full:
	uv run modal run hackathon/egoflow/modal_app.py --action full --episode-ids-file hackathon/egoflow/episode_selection.csv --max-episodes 20 --max-steps 750 --training-runs 1
