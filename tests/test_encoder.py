"""
Tests for the AdvancedYAMLEncoder and YAMLEncoder classes.
"""

import os
import math
import tempfile
import pytest
import yaml

from uq.util.Encoder import YAMLEncoder, AdvancedYAMLEncoder, create_yaml_encoder


@pytest.fixture
def simple_yaml_template(tmp_path):
    """Create a simple YAML template file for testing."""
    content = "model:\n" "  name: test\n" "  value: $PARAM1$\n" "settings:\n" "  tolerance: $TOL$\n"
    template_file = tmp_path / "template.yaml"
    template_file.write_text(content)
    return str(template_file)


@pytest.fixture
def nested_yaml_template(tmp_path):
    """Create a nested YAML template file for testing."""
    config = {
        "materials": [
            {
                "D_0": {"mean": 1.5e-6, "std": 0.1},
                "thermal_conductivity": {"mean": 10.0},
            }
        ],
        "geometry": {
            "domains": [{"length": 0.01}],
        },
        "simulation": {"tolerances": {"absolute_tolerance": {"tritium_transport": 1e10}}},
    }
    template_file = tmp_path / "nested_template.yaml"
    with open(str(template_file), "w") as f:
        yaml.dump(config, f)
    return str(template_file)


class TestYAMLEncoder:
    """Tests for the basic YAMLEncoder."""

    def test_init_valid_template(self, simple_yaml_template):
        enc = YAMLEncoder(simple_yaml_template)
        assert enc.template_fname == simple_yaml_template
        assert enc.target_filename == "config.yaml"
        assert enc.delimiter == "$"

    def test_init_missing_template_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            YAMLEncoder(str(tmp_path / "nonexistent.yaml"))

    def test_encode_substitutes_parameters(self, simple_yaml_template, tmp_path):
        enc = YAMLEncoder(simple_yaml_template, target_filename="out.yaml")
        params = {"PARAM1": "42.0", "TOL": "1e-8"}
        enc.encode(params=params, target_dir=str(tmp_path))

        out_path = tmp_path / "out.yaml"
        assert out_path.exists()
        content = out_path.read_text()
        assert "42.0" in content
        assert "1e-8" in content
        assert "$PARAM1$" not in content

    def test_encode_no_params_leaves_template(self, simple_yaml_template, tmp_path):
        enc = YAMLEncoder(simple_yaml_template, target_filename="out.yaml")
        enc.encode(params={}, target_dir=str(tmp_path))
        content = (tmp_path / "out.yaml").read_text()
        assert "$PARAM1$" in content

    def test_get_restart_dict(self, simple_yaml_template):
        enc = YAMLEncoder(simple_yaml_template, target_filename="test.yaml", delimiter="%")
        rd = enc.get_restart_dict()
        assert rd["template_fname"] == simple_yaml_template
        assert rd["target_filename"] == "test.yaml"
        assert rd["delimiter"] == "%"

    def test_deserialize_roundtrip(self, simple_yaml_template):
        enc = YAMLEncoder(simple_yaml_template, target_filename="t.yaml", delimiter="#")
        rd = enc.get_restart_dict()
        enc2 = YAMLEncoder.deserialize(rd)
        assert enc2.template_fname == enc.template_fname
        assert enc2.target_filename == enc.target_filename
        assert enc2.delimiter == enc.delimiter


class TestAdvancedYAMLEncoder:
    """Tests for the AdvancedYAMLEncoder."""

    def test_init_with_parameter_map(self, nested_yaml_template):
        pmap = {"D_0": "materials.D_0.mean"}
        enc = AdvancedYAMLEncoder(nested_yaml_template, parameter_map=pmap)
        assert enc.parameter_map == pmap

    def test_init_missing_template_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AdvancedYAMLEncoder(str(tmp_path / "missing.yaml"))

    def test_set_nested_value(self, nested_yaml_template):
        enc = AdvancedYAMLEncoder(nested_yaml_template)
        config = {"a": {"b": {"c": 1.0}}}
        enc._set_nested_value(config, "a.b.c", 2.0)
        assert math.isclose(config["a"]["b"]["c"], 2.0)

    def test_set_nested_value_creates_intermediate(self, nested_yaml_template):
        enc = AdvancedYAMLEncoder(nested_yaml_template)
        config = {}
        enc._set_nested_value(config, "x.y.z", 3.14)
        assert math.isclose(config["x"]["y"]["z"], 3.14)

    def test_set_nested_value_with_list(self, nested_yaml_template):
        enc = AdvancedYAMLEncoder(nested_yaml_template)
        config = {"materials": [{"D_0": {"mean": 1.0}}]}
        enc._set_nested_value(config, "materials.D_0.mean", 5.0)
        assert math.isclose(config["materials"][0]["D_0"]["mean"], 5.0)

    def test_set_nested_value_empty_path_raises(self, nested_yaml_template):
        enc = AdvancedYAMLEncoder(nested_yaml_template)
        with pytest.raises(ValueError, match="Path cannot be empty"):
            enc._set_nested_value({}, "", 1.0)

    def test_set_nested_value_invalid_path_type_raises(self, nested_yaml_template):
        enc = AdvancedYAMLEncoder(nested_yaml_template)
        with pytest.raises(ValueError, match="Path must be a string"):
            enc._set_nested_value({}, 123, 1.0)

    def test_encode_with_parameter_map(self, nested_yaml_template, tmp_path):
        pmap = {"D_0": "materials.D_0.mean"}
        enc = AdvancedYAMLEncoder(
            nested_yaml_template,
            target_filename="output.yaml",
            parameter_map=pmap,
            type_conversions={"D_0": float},
        )
        enc.encode(params={"D_0": 2.5e-7}, target_dir=str(tmp_path))

        with open(str(tmp_path / "output.yaml")) as f:
            result = yaml.safe_load(f)

        assert math.isclose(result["materials"][0]["D_0"]["mean"], 2.5e-7)

    def test_encode_recursive_fallback(self, nested_yaml_template, tmp_path):
        enc = AdvancedYAMLEncoder(
            nested_yaml_template,
            target_filename="output.yaml",
            parameter_map={},
        )
        # The _update_config_recursive method should find "mean" at materials[0].D_0.mean
        # but since "mean" is ambiguous, test with a unique key
        # For this test, we add a unique key to the template first
        with open(nested_yaml_template) as f:
            config = yaml.safe_load(f)
        config["unique_param"] = 99.0
        with open(nested_yaml_template, "w") as f:
            yaml.dump(config, f)

        enc2 = AdvancedYAMLEncoder(nested_yaml_template, target_filename="output.yaml")
        enc2.encode(params={"unique_param": 42.0}, target_dir=str(tmp_path))

        with open(str(tmp_path / "output.yaml")) as f:
            result = yaml.safe_load(f)
        assert result["unique_param"] == 42.0

    def test_encode_with_fixed_parameters(self, nested_yaml_template, tmp_path):
        pmap = {
            "D_0": "materials.D_0.mean",
            "fixed_key": "simulation.tolerances.absolute_tolerance.tritium_transport",
        }
        enc = AdvancedYAMLEncoder(
            nested_yaml_template,
            target_filename="output.yaml",
            parameter_map=pmap,
            type_conversions={"D_0": float, "fixed_key": float},
            fixed_parameters={"fixed_key": 1e5},
        )
        enc.encode(params={"D_0": 3.0e-6}, target_dir=str(tmp_path))

        with open(str(tmp_path / "output.yaml")) as f:
            result = yaml.safe_load(f)

        assert math.isclose(
            result["simulation"]["tolerances"]["absolute_tolerance"]["tritium_transport"],
            1e5,
        )

    def test_get_restart_dict(self, nested_yaml_template):
        pmap = {"D_0": "materials.D_0.mean"}
        enc = AdvancedYAMLEncoder(nested_yaml_template, parameter_map=pmap, type_conversions={"D_0": float})
        rd = enc.get_restart_dict()
        assert rd["parameter_map"] == pmap
        assert rd["type_conversions"] == {"D_0": float}

    def test_deserialize_roundtrip(self, nested_yaml_template):
        pmap = {"D_0": "materials.D_0.mean"}
        enc = AdvancedYAMLEncoder(nested_yaml_template, parameter_map=pmap, type_conversions={"D_0": float})
        rd = enc.get_restart_dict()
        enc2 = AdvancedYAMLEncoder.deserialize(rd)
        assert enc2.parameter_map == enc.parameter_map


class TestCreateYamlEncoder:
    """Tests for the create_yaml_encoder convenience function."""

    def test_create_basic(self, simple_yaml_template):
        enc = create_yaml_encoder(simple_yaml_template)
        assert isinstance(enc, YAMLEncoder)
        assert not isinstance(enc, AdvancedYAMLEncoder)

    def test_create_advanced(self, nested_yaml_template):
        enc = create_yaml_encoder(nested_yaml_template, advanced=True)
        assert isinstance(enc, AdvancedYAMLEncoder)
