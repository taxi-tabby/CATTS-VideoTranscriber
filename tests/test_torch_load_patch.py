"""torch.load 패치가 weights_only=False를 강제하는지 검증."""
import io
import pytest


class TestTorchLoadPatch:
    """_patch_torch_load가 모든 torch.load 호출에서 weights_only=False를 강제하는지 테스트."""

    def test_patch_forces_weights_only_false_when_not_specified(self):
        """weights_only 미지정 시 False로 설정되는지 확인."""
        import torch
        from src.torchaudio_compat import _patch_torch_load

        captured = {}
        original = torch.load

        def mock_load(*args, **kwargs):
            captured.update(kwargs)

        try:
            torch.load = mock_load
            torch.load._compat_patched = False
            _patch_torch_load()

            torch.load("dummy_path")
            assert captured.get("weights_only") is False
        finally:
            torch.load = original

    def test_patch_overrides_explicit_weights_only_true(self):
        """weights_only=True가 명시적으로 전달되어도 False로 강제되는지 확인."""
        import torch
        from src.torchaudio_compat import _patch_torch_load

        captured = {}
        original = torch.load

        def mock_load(*args, **kwargs):
            captured.update(kwargs)

        try:
            torch.load = mock_load
            torch.load._compat_patched = False
            _patch_torch_load()

            torch.load("dummy_path", weights_only=True)
            assert captured.get("weights_only") is False
        finally:
            torch.load = original

    def test_patch_overrides_weights_only_none(self):
        """weights_only=None (lightning의 기본 전달값)도 False로 강제되는지 확인."""
        import torch
        from src.torchaudio_compat import _patch_torch_load

        captured = {}
        original = torch.load

        def mock_load(*args, **kwargs):
            captured.update(kwargs)

        try:
            torch.load = mock_load
            torch.load._compat_patched = False
            _patch_torch_load()

            torch.load("dummy_path", weights_only=None)
            assert captured.get("weights_only") is False
        finally:
            torch.load = original

    def test_patch_idempotent(self):
        """여러 번 호출해도 이중 패치되지 않는지 확인."""
        import torch
        from src.torchaudio_compat import _patch_torch_load

        call_count = 0
        original = torch.load

        def mock_load(*args, **kwargs):
            nonlocal call_count
            call_count += 1

        try:
            torch.load = mock_load
            torch.load._compat_patched = False
            _patch_torch_load()
            first_patched = torch.load
            _patch_torch_load()  # 두 번째 호출

            assert torch.load is first_patched
            torch.load("dummy_path")
            assert call_count == 1
        finally:
            torch.load = original

    def test_apply_all_patches_includes_torch_load(self):
        """apply_all_patches()가 torch.load 패치를 포함하는지 확인."""
        import torch
        from src import torchaudio_compat

        original = torch.load

        try:
            torchaudio_compat._patched = False
            torchaudio_compat.apply_all_patches()
            assert getattr(torch.load, "_compat_patched", False) is True
        finally:
            torch.load = original
            torchaudio_compat._patched = False

    def test_real_checkpoint_loads_with_patch(self):
        """실제 pyannote 체크포인트 파일이 패치된 torch.load로 로드되는지 확인."""
        import os
        import torch
        from src import torchaudio_compat

        ckpt = os.path.expanduser(
            "~/.cache/torch/pyannote/models--pyannote--segmentation-3.0/"
            "snapshots/e66f3d3b9eb0873085418a7b813d3b369bf160bb/pytorch_model.bin"
        )
        if not os.path.exists(ckpt):
            pytest.skip("pyannote checkpoint not cached locally")

        original = torch.load
        try:
            torchaudio_compat._patched = False
            torchaudio_compat.apply_all_patches()

            # weights_only=True, =None 모두 성공해야 함
            result = torch.load(ckpt, map_location="cpu", weights_only=True)
            assert "state_dict" in result

            result2 = torch.load(ckpt, map_location="cpu", weights_only=None)
            assert "state_dict" in result2
        finally:
            torch.load = original
            torchaudio_compat._patched = False

    def test_serialization_load_also_patched(self):
        """torch.serialization.load도 패치되어야 한다 — 내부 코드가 직접 호출할 수 있음."""
        import torch
        import torch.serialization
        from src import torchaudio_compat

        original_load = torch.load
        original_ser_load = torch.serialization.load

        try:
            torchaudio_compat._patched = False
            torchaudio_compat.apply_all_patches()

            # torch.serialization.load도 패치되었는지 확인
            assert getattr(torch.serialization.load, "_compat_patched", False) is True

            # 실제로 weights_only=False가 강제되는지 확인
            buf = io.BytesIO()
            torch.save({"test": 42}, buf)
            buf.seek(0)
            result = torch.serialization.load(buf, weights_only=True)
            assert result == {"test": 42}
        finally:
            torch.load = original_load
            torch.serialization.load = original_ser_load
            torchaudio_compat._patched = False
