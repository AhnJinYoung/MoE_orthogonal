## Project Structure & Module Organization

프로젝트 :  MoE 구조에서 top-1 expert를 제외한 나머지 expert의 orthogonal 한 성분만 더하여 MoE구조의 표현력을 높이는 실험

- `READMD.md` 실험의 모든 과정이 디테일하게 기록되어있음.
- `run.sh` 실험 run을 총괄하는 스크립트. 여러 옵션을 줘서 다르게 실험을 돌릴 수 있음.
- `run.md` run.sh의 옵션을 정리한 .md파일. 각 실험별로 어떻게 돌려야 하는지 예시가 있음
- `config/` 다양한 모델, 다양한 옵션에서 실험을 돌릴 수 있도록 다양한 profile을 담은 config파일.
- `src/` 실험코드. 되도록 적은 파일 수를 지향함. 기존 모델에 orthogonal sum을 적용하는 실험과 pretrian부터 적용하는 두 가지 경우의 코드가 있음.


## Coding Style & Naming Conventions

Use clear, conventional names. For Python, prefer `snake_case` for modules, functions, variables, and test files; use `PascalCase` for classes and `UPPER_SNAKE_CASE` for constants.

Use 4-space indentation for Python and keep functions focused. Favor explicit arguments and small helpers over hidden global state, especially for experiment configuration.


## Rules for editing codes

변경사항이 발생하면 코드가 정상적으로 작동할 수 있는지 항상 검증할 것. 

사용자가 요구하는 구현의 범위를 명확하게 하고 만약 모호하다면 다시 질문해서 구현의 범위를 명확하게 할 것. 

## Commit & Pull Request Guidelines

git url : https://github.com/AhnJinYoung/MoE_orthogonal.git (public)

git ignore : every api key, this agents.md file.

모든 변경사항이 발생했을 때 git push. 

## Security & Configuration Tips

Do not commit secrets, API keys, private datasets, or machine-specific paths. Use environment variables or local ignored config files for sensitive settings.
