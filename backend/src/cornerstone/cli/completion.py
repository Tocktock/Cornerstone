from __future__ import annotations


def completion_script(shell: str) -> str:
    shell = shell.lower()
    if shell == "bash":
        return """# Cornerstone bash completion
_cornerstone_completion() {
  local cur
  cur="${COMP_WORDS[COMP_CWORD]}"
  COMPREPLY=( $(compgen -W "version doctor setup local env stack db api worker live status source evidence concept ask context eval proof config completion" -- "$cur") )
}
complete -F _cornerstone_completion cornerstone
"""
    if shell == "zsh":
        return """#compdef cornerstone
_arguments '1:command:(version doctor setup local env stack db api worker live status source evidence concept ask context eval proof config completion)'
"""
    if shell == "powershell":
        return """Register-ArgumentCompleter -Native -CommandName cornerstone -ScriptBlock {
  param($wordToComplete, $commandAst, $cursorPosition)
  'version','doctor','setup','local','env','stack','db','api','worker','live','status','source','evidence','concept','ask','context','eval','proof','config','completion' |
    Where-Object { $_ -like "$wordToComplete*" } |
    ForEach-Object { [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_) }
}
"""
    raise RuntimeError(f"Unsupported shell: {shell}")
