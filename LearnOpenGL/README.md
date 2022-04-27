`launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
      {
        "name": "cl.exe build and debug active file", // 跟tasks.json的"label"保持一致
        "type": "cppvsdbg",
        "request": "launch",
        "program": "${workspaceFolder}\\build\\src\\02_Hello_Triangle\\Debug\\${fileBasenameNoExtension}.exe", // Debug需要更改此处的.exe路径
        "args": [],
        "stopAtEntry": false,
        "cwd": "${workspaceFolder}",
        "environment": [],
        "externalConsole": false,
        "preLaunchTask": "cl.exe build active file"
      }
    ]
  }
```

`tasks.json`:

```json
{
    "tasks": [
        {
            "type": "cppbuild",
            "label": "cl.exe build active file", // 跟launch.json的"name"保持一致
            "command": "cl.exe",
            "args": [
                "/Zi",
                "/EHsc",
                "/nologo",
                "/Fe:",
                "${workspaceFolder}\\build\\src\\02_Hello_Triangle\\Debug\\${fileBasenameNoExtension}.exe", // 跟launch.json的"program"保持一致
                "${file}"
            ],
            "options": {
                "cwd": "${workspaceFolder}"
            },
            "problemMatcher": [
                "$msCompile"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "detail": "调试器生成的任务。"
        }
    ],
    "version": "2.0.0"
}
```