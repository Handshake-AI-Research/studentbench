"""Interrupted exports resume only artifacts whose dependencies still match."""

from studentbench.render_checkpoints import RenderCheckpoints


def test_resume_detects_changed_inputs_and_tampered_output(tmp_path):
    root, output = tmp_path / 'repo', tmp_path / 'results'
    root.mkdir()
    (root / 'reproduce.py').write_text('# driver')
    (root / 'pyproject.toml').write_text('')
    source = tmp_path / 'estimates.csv'
    source.write_text('estimate\n1.5\n')
    artifact = tmp_path / 'figure.pdf'
    builds = []

    def render():
        builds.append(True)
        artifact.write_bytes(source.read_bytes())

    for _ in range(2):
        RenderCheckpoints(root, output).run('figure', [source], [artifact], render)
    assert len(builds) == 1
    artifact.write_text('incomplete')
    RenderCheckpoints(root, output).run('figure', [source], [artifact], render)
    assert len(builds) == 2
    source.write_text('estimate\n2.5\n')
    RenderCheckpoints(root, output).run('figure', [source], [artifact], render)
    assert len(builds) == 3
    (root / 'reproduce.py').write_text('# updated renderer')
    RenderCheckpoints(root, output).run('figure', [source], [artifact], render)
    assert len(builds) == 4


def test_resume_rebuilds_when_fonts_or_renderer_change(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from matplotlib import font_manager
    import studentbench.render_checkpoints as checkpoints

    root, output = tmp_path / 'repo', tmp_path / 'results'
    root.mkdir()
    (root / 'reproduce.py').write_text('# driver')
    (root / 'pyproject.toml').write_text('')
    font = tmp_path / 'font.ttf'
    font.write_bytes(b'fallback font')
    monkeypatch.setattr(font_manager.fontManager, 'ttflist', [])
    monkeypatch.setattr(font_manager, 'findfont', lambda *args, **kwargs: str(font))
    artifact = tmp_path / 'figure.pdf'
    builds = []

    def render():
        builds.append(True)
        artifact.write_bytes(b'figure')

    def resume():
        RenderCheckpoints(root, output).run('figure', [], [artifact], render)

    resume()
    resume()
    assert len(builds) == 1
    # Installing Arial changes the chosen font even if its bytes happen to match.
    monkeypatch.setattr(font_manager.fontManager, 'ttflist', [SimpleNamespace(name='Arial')])
    resume()
    assert len(builds) == 2
    font.write_bytes(b'updated Arial font')
    resume()
    assert len(builds) == 3
    original_version = checkpoints.version
    monkeypatch.setattr(checkpoints, 'version',
                        lambda name: 'changed' if name == 'PyMuPDF' else original_version(name))
    resume()
    assert len(builds) == 4
    monkeypatch.setattr(checkpoints.platform, 'python_version', lambda: 'changed')
    resume()
    assert len(builds) == 5
    monkeypatch.setitem(sys.modules, 'cairocffi',
                        SimpleNamespace(cairo_version_string=lambda: 'changed'))
    resume()
    assert len(builds) == 6
    resume()
    assert len(builds) == 6
