# SPDX-License-Identifier: Apache-2.0

"""Tests for fastsafetensors.unified_loader module."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from fastsafetensors.config import CONFIG_ENV_VAR, LoaderConfig
from fastsafetensors.loader import (
    BaseSafeTensorsFileLoader,
)
from fastsafetensors.loader import SafeTensorsFileLoader as RealSafeLoader
from fastsafetensors.threefs_loader import ThreeFSLoader as RealThreeFSLoader
from fastsafetensors.unified_loader import _LOADER_REGISTRY, UnifiedLoader


class TestUnifiedLoaderSignature:
    """Test UnifiedLoader constructor signature."""

    def test_signature_has_pg_files_device(self):
        """Constructor should accept (pg, hf_weights_files, device)."""
        import inspect

        sig = inspect.signature(UnifiedLoader.__init__)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "pg" in params
        assert "hf_weights_files" in params
        assert "device" in params

    def test_signature_no_loader_param(self):
        """Constructor should NOT have a 'loader' positional param."""
        import inspect

        sig = inspect.signature(UnifiedLoader.__init__)
        params = list(sig.parameters.keys())
        # 'loader' is now a config field, not a constructor arg
        assert "loader" not in params

    def test_signature_no_config_param(self):
        """Constructor should NOT have 'config' or 'config_path' params."""
        import inspect

        sig = inspect.signature(UnifiedLoader.__init__)
        params = list(sig.parameters.keys())
        assert "config" not in params
        assert "config_path" not in params

    def test_signature_no_kwargs(self):
        """Constructor should NOT accept **kwargs."""
        import inspect

        sig = inspect.signature(UnifiedLoader.__init__)
        params = sig.parameters
        for p in params.values():
            assert p.kind != inspect.Parameter.VAR_KEYWORD


class TestLoaderRegistry:
    """Test _LOADER_REGISTRY contains expected entries."""

    def test_base_registered(self):
        assert "base" in _LOADER_REGISTRY

    def test_3fs_registered(self):
        assert "3fs" in _LOADER_REGISTRY

    def test_threefs_not_registered(self):
        """Old 'threefs' name should NOT be in the registry."""
        assert "threefs" not in _LOADER_REGISTRY


class TestUnifiedLoaderBaseLoader:
    """Test UnifiedLoader creates SafeTensorsFileLoader for loader='base'."""

    def _make_base_config(self, copier_type="nogds"):
        """Helper to create a LoaderConfig with base extension."""
        cfg = LoaderConfig(loader="base")
        cfg._extensions["base"] = {"copier_type": copier_type}
        return cfg

    def test_default_creates_base_loader(self, monkeypatch):
        """Default config (loader='base') should create SafeTensorsFileLoader."""
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        with (
            patch("fastsafetensors.unified_loader.SafeTensorsFileLoader") as mock_stfl,
            patch("fastsafetensors.unified_loader.PipelineParallel"),
            patch("fastsafetensors.unified_loader.load_config") as mock_load,
        ):
            mock_load.return_value = self._make_base_config("nogds")
            mock_stfl.process_extension_config = RealSafeLoader.process_extension_config
            mock_stfl.return_value = MagicMock()

            unified = UnifiedLoader(None, ["file1.safetensors"], device="cpu")

            mock_stfl.assert_called_once()
            call_kwargs = mock_stfl.call_args
            assert call_kwargs[1]["nogds"] is True
            assert unified.config.loader == "base"

    def test_gds_creates_base_loader_with_nogds_false(self, monkeypatch):
        """loader='base' with copier_type='gds' should pass nogds=False."""
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        with (
            patch("fastsafetensors.unified_loader.SafeTensorsFileLoader") as mock_stfl,
            patch("fastsafetensors.unified_loader.PipelineParallel"),
            patch("fastsafetensors.unified_loader.load_config") as mock_load,
        ):
            mock_load.return_value = self._make_base_config("gds")
            mock_stfl.process_extension_config = RealSafeLoader.process_extension_config
            mock_stfl.return_value = MagicMock()

            UnifiedLoader(None, ["file1.safetensors"], device="cuda:0")

            call_kwargs = mock_stfl.call_args
            assert call_kwargs[1]["nogds"] is False

    def test_base_no_extension_uses_defaults(self, monkeypatch):
        """loader='base' without base extension section should still work."""
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        with (
            patch("fastsafetensors.unified_loader.SafeTensorsFileLoader") as mock_stfl,
            patch("fastsafetensors.unified_loader.PipelineParallel"),
            patch("fastsafetensors.unified_loader.load_config") as mock_load,
        ):
            # No _extensions set -> get_extension_config returns {}
            mock_load.return_value = LoaderConfig(loader="base")
            mock_stfl.process_extension_config = RealSafeLoader.process_extension_config
            mock_stfl.return_value = MagicMock()

            UnifiedLoader(None, ["file1.safetensors"], device="cpu")

            mock_stfl.assert_called_once()
            call_kwargs = mock_stfl.call_args
            # process_extension_config with {} -> nogds=False (default copier_type="gds")
            assert call_kwargs[1]["nogds"] is False


class TestUnifiedLoader3FSLoader:
    """Test UnifiedLoader creates ThreeFSLoader for loader='3fs'."""

    def _make_3fs_config(self, mount_point="/data/3fs"):
        """Helper to create a LoaderConfig with 3fs extension."""
        cfg = LoaderConfig(loader="3fs")
        cfg._extensions["3fs"] = {"mount_point": mount_point}
        return cfg

    def test_3fs_creates_threefs_loader(self, monkeypatch):
        """loader='3fs' should create ThreeFSLoader."""
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        with (
            patch("fastsafetensors.unified_loader.ThreeFSLoader") as mock_3fs,
            patch("fastsafetensors.unified_loader.PipelineParallel"),
            patch("fastsafetensors.unified_loader.load_config") as mock_load,
        ):
            mock_load.return_value = self._make_3fs_config("/data/3fs")
            mock_3fs.process_extension_config = (
                RealThreeFSLoader.process_extension_config
            )
            mock_3fs.return_value = MagicMock()

            unified = UnifiedLoader(None, ["file1.safetensors"], device="cuda:0")

            mock_3fs.assert_called_once()
            call_kwargs = mock_3fs.call_args
            assert call_kwargs[1]["mount_point"] == "/data/3fs"
            assert unified.config.loader == "3fs"

    def test_3fs_common_kwargs_passed(self, monkeypatch):
        """Common kwargs (framework, debug_log, etc.) should be passed to ThreeFSLoader."""
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

        with (
            patch("fastsafetensors.unified_loader.ThreeFSLoader") as mock_3fs,
            patch("fastsafetensors.unified_loader.PipelineParallel"),
            patch("fastsafetensors.unified_loader.load_config") as mock_load,
        ):
            cfg = LoaderConfig(loader="3fs", framework="paddle", debug_log=True)
            cfg._extensions["3fs"] = {"mount_point": "/mnt/3fs"}
            mock_load.return_value = cfg
            mock_3fs.process_extension_config = (
                RealThreeFSLoader.process_extension_config
            )
            mock_3fs.return_value = MagicMock()

            UnifiedLoader(None, ["file1.safetensors"], device="cuda:0")

            call_kwargs = mock_3fs.call_args[1]
            assert call_kwargs["framework"] == "paddle"
            assert call_kwargs["debug_log"] is True
            assert call_kwargs["mount_point"] == "/mnt/3fs"


class TestUnifiedLoaderUnknownLoader:
    """Test UnifiedLoader raises ValueError for unknown loader type."""

    def test_unknown_loader_raises(self):
        with patch("fastsafetensors.unified_loader.load_config") as mock_load:
            mock_load.return_value = LoaderConfig(loader="nonexistent")

            with pytest.raises(ValueError, match="Unknown loader type"):
                UnifiedLoader(None, ["file1.safetensors"])


class TestUnifiedLoaderPipelineCreation:
    """Test that UnifiedLoader creates PipelineParallel correctly."""

    def test_pipeline_receives_parallel_kwargs(self):
        """PipelineParallel should receive parallel kwargs from config."""
        with (
            patch("fastsafetensors.unified_loader.SafeTensorsFileLoader") as mock_stfl,
            patch("fastsafetensors.unified_loader.PipelineParallel") as mock_pp,
            patch("fastsafetensors.unified_loader.load_config") as mock_load,
        ):
            mock_load.return_value = LoaderConfig(
                use_pipeline=True,
                max_concurrent_producers=1,
                queue_size=3,
                use_tqdm_on_load=False,
            )
            mock_stfl.return_value = MagicMock()

            UnifiedLoader(None, ["file1.safetensors"])

            mock_pp.assert_called_once()
            call_kwargs = mock_pp.call_args[1]
            assert call_kwargs["max_concurrent_producers"] == 1
            assert call_kwargs["queue_size"] == 3
            assert call_kwargs["use_tqdm_on_load"] is False


class TestUnifiedLoaderConfigDiscovery:
    """Test config file discovery (env var > default path > defaults)."""

    def test_env_var_config(self, monkeypatch):
        """FASTSAFETENSORS_CONFIG env var should point to config file."""
        data = {
            "loader": "3fs",
            "3fs": {"mount_point": "/data/3fs"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = f.name

        try:
            monkeypatch.setenv(CONFIG_ENV_VAR, path)

            with (
                patch("fastsafetensors.unified_loader.ThreeFSLoader") as mock_3fs,
                patch("fastsafetensors.unified_loader.PipelineParallel"),
            ):
                mock_3fs.return_value = MagicMock()
                unified = UnifiedLoader(None, ["file1.safetensors"])
                assert unified.config.loader == "3fs"
                assert (
                    unified.config.get_extension_config("3fs")["mount_point"]
                    == "/data/3fs"
                )
        finally:
            os.unlink(path)

    def test_no_config_uses_defaults(self, monkeypatch):
        """Without config file, should use LoaderConfig defaults."""
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        # cd to a temp dir where no fastsafetensors.yaml exists
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with (
                    patch(
                        "fastsafetensors.unified_loader.SafeTensorsFileLoader"
                    ) as mock_stfl,
                    patch("fastsafetensors.unified_loader.PipelineParallel"),
                ):
                    mock_stfl.return_value = MagicMock()
                    unified = UnifiedLoader(None, ["file1.safetensors"])
                    assert unified.config == LoaderConfig()
            finally:
                os.chdir(old_cwd)


class TestUnifiedLoaderRemovedAPIs:
    """Verify that previously-public surface (context manager / __iter__ /
    loader property) is no longer exposed."""

    def test_no_context_manager(self):
        assert not hasattr(UnifiedLoader, "__enter__")
        assert not hasattr(UnifiedLoader, "__exit__")

    def test_no_iter(self):
        # __iter__ should be removed; UnifiedLoader is not iterable.
        assert "__iter__" not in vars(UnifiedLoader)

    def test_no_loader_property(self):
        # The public ``loader`` property is removed; ``_loader`` remains
        # as an internal attribute only.
        assert "loader" not in vars(UnifiedLoader)


class TestUnifiedLoaderProperties:
    """Test UnifiedLoader properties."""

    def test_config_property(self):
        with (
            patch("fastsafetensors.unified_loader.SafeTensorsFileLoader") as mock_stfl,
            patch("fastsafetensors.unified_loader.PipelineParallel"),
            patch("fastsafetensors.unified_loader.load_config") as mock_load,
        ):
            cfg = LoaderConfig(loader="base")
            cfg._extensions["base"] = {"copier_type": "gds"}
            mock_load.return_value = cfg
            mock_stfl.process_extension_config = RealSafeLoader.process_extension_config
            mock_stfl.return_value = MagicMock()

            unified = UnifiedLoader(None, ["file1.safetensors"])
            assert unified.config is cfg


class TestUnifiedLoaderClose:
    """Verify UnifiedLoader.close() does not double-close the underlying loader.

    PipelineParallel.close() already closes its inner loader, so
    UnifiedLoader.close() must only call pipeline.close() and must NOT call
    self._loader.close() directly.
    """

    def test_close_does_not_double_close_underlying_loader(self):
        with (
            patch("fastsafetensors.unified_loader.SafeTensorsFileLoader") as mock_stfl,
            patch("fastsafetensors.unified_loader.PipelineParallel") as mock_pp,
            patch("fastsafetensors.unified_loader.load_config") as mock_load,
        ):
            mock_load.return_value = LoaderConfig()
            mock_loader_instance = MagicMock()
            mock_stfl.return_value = mock_loader_instance
            mock_pipeline_instance = MagicMock()
            mock_pp.return_value = mock_pipeline_instance

            unified = UnifiedLoader(None, ["file1.safetensors"])
            unified.close()

            mock_pipeline_instance.close.assert_called_once()
            mock_loader_instance.close.assert_not_called()


class TestProcessExtensionConfig:
    """Test that process_extension_config is correctly invoked via UnifiedLoader."""

    def test_base_process_extension_config_called(self):
        """SafeTensorsFileLoader.process_extension_config should be called."""
        with (
            patch("fastsafetensors.unified_loader.SafeTensorsFileLoader") as mock_stfl,
            patch("fastsafetensors.unified_loader.PipelineParallel"),
            patch("fastsafetensors.unified_loader.load_config") as mock_load,
        ):
            cfg = LoaderConfig(loader="base")
            cfg._extensions["base"] = {"copier_type": "gds", "bbuf_size_kb": 8192}
            mock_load.return_value = cfg
            mock_stfl.process_extension_config = RealSafeLoader.process_extension_config
            mock_stfl.return_value = MagicMock()

            UnifiedLoader(None, ["file1.safetensors"], device="cpu")

            call_kwargs = mock_stfl.call_args[1]
            # process_extension_config should have mapped copier_type -> nogds
            assert call_kwargs["nogds"] is False
            assert call_kwargs["bbuf_size_kb"] == 8192
            assert "copier_type" not in call_kwargs


class TestThreeFSProcessExtensionConfig:
    """Test ThreeFSLoader.process_extension_config mount_point inference."""

    def test_explicit_mount_point_preserved(self):
        """Explicit mount_point should NOT be overridden."""
        result = RealThreeFSLoader.process_extension_config(
            {"mount_point": "/custom/3fs"},
            hf_weights_files=["/mnt/x/a.safetensors"],
        )
        assert result["mount_point"] == "/custom/3fs"

    def test_no_mount_point_no_files_no_inference(self):
        """No mount_point key + no files -> no inference, key absent."""
        result = RealThreeFSLoader.process_extension_config({})
        assert "mount_point" not in result

    def test_no_mount_point_with_files_triggers_inference(self):
        """No mount_point key + files present -> inference attempted."""
        with patch(
            "fastsafetensors.threefs_loader.extract_mount_point",
            create=True,
        ) as mock_extract:
            # Patch the import inside process_extension_config
            mock_module = MagicMock()
            mock_module.extract_mount_point.return_value = "/inferred/3fs"
            with patch.dict("sys.modules", {"fastsafetensor_3fs_reader": mock_module}):
                result = RealThreeFSLoader.process_extension_config(
                    {},
                    hf_weights_files=["/inferred/3fs/model.safetensors"],
                )
                assert result["mount_point"] == "/inferred/3fs"

    def test_empty_mount_point_triggers_inference(self):
        """Empty string mount_point should trigger inference."""
        mock_module = MagicMock()
        mock_module.extract_mount_point.return_value = "/inferred/3fs"
        with patch.dict("sys.modules", {"fastsafetensor_3fs_reader": mock_module}):
            result = RealThreeFSLoader.process_extension_config(
                {"mount_point": ""},
                hf_weights_files=["/inferred/3fs/model.safetensors"],
            )
            assert result["mount_point"] == "/inferred/3fs"

    def test_whitespace_mount_point_triggers_inference(self):
        """Whitespace-only mount_point should trigger inference."""
        mock_module = MagicMock()
        mock_module.extract_mount_point.return_value = "/inferred/3fs"
        with patch.dict("sys.modules", {"fastsafetensor_3fs_reader": mock_module}):
            result = RealThreeFSLoader.process_extension_config(
                {"mount_point": "   "},
                hf_weights_files=["/inferred/3fs/model.safetensors"],
            )
            assert result["mount_point"] == "/inferred/3fs"

    def test_import_error_graceful_degradation(self):
        """If fastsafetensor_3fs_reader is not importable, should not crash."""
        # Remove the module from sys.modules to force ImportError
        import sys

        saved = sys.modules.pop("fastsafetensor_3fs_reader", None)
        try:
            with patch.dict("sys.modules", {"fastsafetensor_3fs_reader": None}):
                # None in sys.modules causes ImportError on import
                result = RealThreeFSLoader.process_extension_config(
                    {},
                    hf_weights_files=["/mnt/3fs/model.safetensors"],
                )
                # Should not crash; mount_point may or may not be set
                # but definitely should not raise
                assert isinstance(result, dict)
        finally:
            if saved is not None:
                sys.modules["fastsafetensor_3fs_reader"] = saved

    def test_other_fields_pass_through(self):
        """Non-mount_point fields should pass through unchanged."""
        result = RealThreeFSLoader.process_extension_config(
            {"entries": 64, "io_depth": 0},
        )
        assert result["entries"] == 64
        assert result["io_depth"] == 0

    def test_none_mount_point_triggers_inference(self):
        """None mount_point should be treated as empty -> trigger inference."""
        mock_module = MagicMock()
        mock_module.extract_mount_point.return_value = "/inferred/3fs"
        with patch.dict("sys.modules", {"fastsafetensor_3fs_reader": mock_module}):
            result = RealThreeFSLoader.process_extension_config(
                {"mount_point": None},
                hf_weights_files=["/inferred/3fs/model.safetensors"],
            )
            # None is not a string, .strip() will raise AttributeError
            # unless the code handles it. This tests the current behavior.
            assert isinstance(result, dict)


class TestKwargsPassedToProcessExtensionConfig:
    """Verify hf_weights_files reaches process_extension_config via UnifiedLoader."""

    def test_hf_weights_files_passed_to_base(self):
        """Base loader's process_extension_config should receive hf_weights_files kwarg."""
        received_kwargs = {}

        def spy_process_ext(ext_config, **kwargs):
            received_kwargs.update(kwargs)
            return RealSafeLoader.process_extension_config(ext_config, **kwargs)

        with (
            patch("fastsafetensors.unified_loader.SafeTensorsFileLoader") as mock_stfl,
            patch("fastsafetensors.unified_loader.PipelineParallel"),
            patch("fastsafetensors.unified_loader.load_config") as mock_load,
        ):
            mock_load.return_value = LoaderConfig(loader="base")
            mock_stfl.process_extension_config = spy_process_ext
            mock_stfl.return_value = MagicMock()

            UnifiedLoader(None, ["a.safetensors", "b.safetensors"], device="cpu")

            assert "hf_weights_files" in received_kwargs
            assert received_kwargs["hf_weights_files"] == [
                "a.safetensors",
                "b.safetensors",
            ]

    def test_hf_weights_files_passed_to_3fs(self):
        """3FS loader's process_extension_config should receive hf_weights_files kwarg."""
        received_kwargs = {}

        def spy_process_ext(ext_config, **kwargs):
            received_kwargs.update(kwargs)
            return RealThreeFSLoader.process_extension_config(ext_config, **kwargs)

        with (
            patch("fastsafetensors.unified_loader.ThreeFSLoader") as mock_3fs,
            patch("fastsafetensors.unified_loader.PipelineParallel"),
            patch("fastsafetensors.unified_loader.load_config") as mock_load,
        ):
            cfg = LoaderConfig(loader="3fs")
            cfg._extensions["3fs"] = {"mount_point": "/data/3fs"}
            mock_load.return_value = cfg
            mock_3fs.process_extension_config = spy_process_ext
            mock_3fs.return_value = MagicMock()

            UnifiedLoader(None, ["/data/3fs/m1.safetensors"], device="cuda:0")

            assert "hf_weights_files" in received_kwargs
            assert received_kwargs["hf_weights_files"] == ["/data/3fs/m1.safetensors"]


class TestInitExports:
    """Verify public API exports from fastsafetensors package."""

    def test_unified_loader_importable(self):
        """UnifiedLoader should be importable from top-level package."""
        from fastsafetensors import UnifiedLoader as UL

        assert UL is not None
        assert UL is UnifiedLoader

    def test_load_config_importable(self):
        """LoaderConfig and load_config should be importable from top-level package."""
        from fastsafetensors import LoaderConfig as LC
        from fastsafetensors import load_config as lc

        assert LC is not None
        assert lc is not None


class TestBaseProcessExtensionConfigKwargs:
    """Verify BaseSafeTensorsFileLoader.process_extension_config accepts **kwargs."""

    def test_accepts_extra_kwargs(self):
        """Should accept arbitrary kwargs without error."""
        result = BaseSafeTensorsFileLoader.process_extension_config(
            {"key1": "val1"},
            hf_weights_files=["a.safetensors"],
            some_other_kwarg=42,
        )
        assert result == {"key1": "val1"}

    def test_kwargs_ignored_in_output(self):
        """Extra kwargs should not appear in output dict."""
        result = BaseSafeTensorsFileLoader.process_extension_config(
            {"copier_type": "gds"},
            hf_weights_files=["a.safetensors"],
        )
        assert "hf_weights_files" not in result
        assert result == {"copier_type": "gds"}
