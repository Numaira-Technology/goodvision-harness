"""Exercise Windows image-update discovery without touching a Docker daemon."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(os.name != 'nt', reason='Windows PowerShell entrypoint')
@pytest.mark.parametrize('shell', ['powershell.exe', 'pwsh'])
@pytest.mark.parametrize('case', [
    'customer', 'explicit', 'wrong_project', 'multiple_projects', 'mixed_images',
    'worker_only', 'backups_helpers', 'missing_selected', 'null_labels',
])
def test_image_discovery_is_explicit_and_keeps_selection_guards(tmp_path, shell, case):
    if not shutil.which(shell):
        pytest.skip(f'{shell} unavailable')
    source = (Path(__file__).resolve().parents[1] / 'tools/harness_update/update_image.ps1').read_text('utf-8-sig')
    source = source.split('try { Start-HarnessUpdate }')[0]
    items = []
    for project, services in (
        ('goodvision-harness', ['harness', 'harness-02', 'rich-label-worker']),
        ('goodvision-contour', ['contour', 'engine']),
    ):
        for service in services:
            items.append({
                'Id': f'id{len(items)}', 'Name': f'/{project}-{service}-1',
                'Image': 'legacy-image', 'Config': {'Labels': {
                    'com.docker.compose.project': project,
                    'com.docker.compose.service': service,
                    'goodvision.ui.release': '20260930-legacy-gpuhost1',
                    'com.docker.compose.project.working_dir': r'C:\Users\Lemon\Desktop\goodvision\goodvision-full-20260923-full1',
                }},
            })
    options = []
    expected_error = None
    names = ','.join(item['Name'].lstrip('/') for item in items[:2])
    if case == 'explicit':
        options = ['-Project', 'goodvision-harness', '-Containers', names]
    elif case == 'wrong_project':
        options = ['-Project', 'different-project']
        expected_error = 'No GoodVision main application matched'
    elif case == 'multiple_projects':
        items[1]['Config']['Labels']['com.docker.compose.project'] = 'another-installation'
        expected_error = 'Multiple installations found'
    elif case == 'mixed_images':
        items[1]['Image'] = 'another-image'
        expected_error = 'mixed versions'
    elif case == 'worker_only':
        items = items[2:]
        expected_error = 'No GoodVision main application matched'
    elif case == 'backups_helpers':
        backup = copy.deepcopy(items[0])
        backup.update(Id='backup', Name='/goodvision-harness-harness-1-before-upgrade')
        helper = copy.deepcopy(items[0])
        helper.update(Id='helper', Name='/temporary-helper')
        helper['Config']['Labels']['goodvision.update.helper'] = 'true'
        items.extend([backup, helper])
    elif case == 'missing_selected':
        options = ['-Containers', names + ',missing-container']
        expected_error = 'selected application containers were not found'
    elif case == 'null_labels':
        items.append({'Id': 'unlabelled', 'Name': '/unrelated', 'Image': 'other', 'Config': {'Labels': None}})
    (tmp_path / 'containers.json').write_text(json.dumps(items), encoding='utf-8')
    source += r'''
function Invoke-UpdateDocker {
    param([string[]]$Arguments, [switch]$Capture)
    $joined = $Arguments -join '|'
    if ($joined -eq 'context|show') { return 'desktop-linux' }
    $fixtures = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'containers.json') | ConvertFrom-Json
    if ($joined -eq 'container|ls|--all|--quiet') { return (($fixtures | ForEach-Object { $_.Id }) -join "`n") }
    if ($Arguments.Count -eq 5 -and ($Arguments[0..3] -join '|') -eq 'container|inspect|--format|{{json .}}') {
        $id = $Arguments[4]
        $found = @($fixtures | Where-Object { $_.Id -eq $id })
        if ($found.Count -ne 1) { throw 'Unexpected container ID' }
        return ($found[0] | ConvertTo-Json -Depth 20 -Compress)
    }
    throw ('Unexpected Docker call: ' + $joined)
}
$manifest = [pscustomobject]@{windows_base_ids=@()}
$selection = Find-InstalledHarness $manifest $PSScriptRoot
Write-Output ('RESULT:' + ($selection | ConvertTo-Json -Compress))
'''
    script = tmp_path / 'discovery.ps1'
    script.write_text(source, encoding='utf-8-sig')
    result = subprocess.run([shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script), *options], capture_output=True, text=True, errors='replace', timeout=30)
    output = result.stdout + result.stderr
    if expected_error:
        assert result.returncode != 0, output
        assert expected_error in output, output
        assert not (tmp_path / 'selection.json').exists()
    else:
        assert result.returncode == 0, output
        assert 'Main application matches: 2' in output
        selected_lines = [line for line in result.stdout.splitlines() if line.startswith('Container ') and line.endswith('| selected')]
        assert len(selected_lines) == 2
        assert all(item['Name'].lstrip('/') in '\n'.join(selected_lines) for item in items[:2])
        selection = json.loads(next(line[7:] for line in result.stdout.splitlines() if line.startswith('RESULT:')))
        assert selection['project'] == 'goodvision-harness'
        assert selection['base_id'] == 'legacy-image'
        assert selection['containers'] == (names if case == 'explicit' else '')
