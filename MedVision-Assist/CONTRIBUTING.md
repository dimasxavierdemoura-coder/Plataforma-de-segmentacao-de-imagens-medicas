# Guia de Contribuição

Obrigado por considerar contribuir para o MedVision Assist! Este documento fornece diretrizes para contribuir ao projeto.

## Como Contribuir

### 1. Fork e Clone

```bash
# Fork o repositório no GitHub
# Clone sua cópia
git clone https://github.com/seu-usuario/MedVision-Assist.git
cd MedVision-Assist
```

### 2. Crie uma Branch

```bash
# Crie uma branch com nome descritivo
git checkout -b feature/sua-feature-aqui
# ou
git checkout -b fix/seu-bugfix-aqui
```

### 3. Desenvolva com Testes

```bash
# Instale dependências
pip install -r requirements.txt

# Desenvolva sua feature
# Adicione testes em tests/

# Execute testes localmente
python -m pytest tests/ -v

# Verificar cobertura
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### 4. Commit e Push

```bash
# Commits claros e descritivos
git add .
git commit -m "feature: adiciona suporte a 3D visualization"

# Push para sua branch
git push origin feature/sua-feature-aqui
```

### 5. Abra um Pull Request

No GitHub:
1. Descreva clara e concisamente a mudança
2. Referencie issues relacionadas (se houver)
3. Aguarde revisão e feedback

## Padrões de Código

### Python Style

Seguimos [PEP 8](https://www.python.org/dev/peps/pep-0008/):

```bash
# Verificar estilo (opcional, recomendado)
pip install flake8
flake8 src/ tests/
```

### Type Hints

Use type hints em todas as funções:

```python
def predict_volume(
    model: torch.nn.Module,
    image: np.ndarray,
    device: str = "cuda"
) -> np.ndarray:
    """Segmentação volumétrica de imagem médica."""
    pass
```

### Docstrings

Use docstrings em formato Google:

```python
def dice_coefficient(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6
) -> float:
    """Calcula coeficiente de Dice entre predição e target.
    
    Args:
        pred: Array binário de predição (N, H, W, D)
        target: Array binário de target (N, H, W, D)
        smooth: Fator de suavização para evitar divisão por zero
        
    Returns:
        float: Valor de Dice (0-1)
        
    Raises:
        ValueError: Se shapes não correspondem
    """
```

## Testes

### Adicionar Testes

Para nova feature, adicione testes correspondentes:

```python
# tests/test_new_feature.py
import pytest
from src.your_module import your_function

def test_your_function():
    """Testa comportamento esperado."""
    result = your_function(input_data)
    assert result == expected_output

def test_your_function_edge_case():
    """Testa caso extremo."""
    with pytest.raises(ValueError):
        your_function(invalid_input)
```

### Cobertura Mínima

- Novos módulos: >80% cobertura
- Modificações: >90% das linhas alteradas

## Branches e Commits

### Naming Convention

```
feature/descriptive-feature-name
fix/descriptive-bug-fix
docs/descriptive-documentation-update
refactor/descriptive-refactoring
perf/descriptive-performance-improvement
test/descriptive-test-addition
```

### Commit Messages

```
type(scope): brief description

Longer explanation if needed. Explain the motivation
and any relevant context.

Fixes #123
```

Tipos válidos:
- `feat`: Nova feature
- `fix`: Correção de bug
- `docs`: Documentação
- `style`: Formatação
- `refactor`: Refatoração sem mudança de comportamento
- `perf`: Melhoria de performance
- `test`: Adição/modificação de testes

## Processo de Review

1. **CI/CD deve passar**: Todos os testes devem passar
2. **Code Review**: Revisor verificará qualidade do código
3. **Feedback**: Responda aos comentários de revisão
4. **Merge**: Após aprovação, a branch será merged

## Comunicação

- Use GitHub Issues para relatar bugs ou sugerir features
- Descreva o contexto claro e exemplos reproduzíveis
- Seja respeitoso e construtivo

## Questões ou Ajuda?

- Consulte a documentação em [README.md](README.md)
- Veja [DATASET_SETUP.md](DATASET_SETUP.md) para configuração
- Abra uma issue para dúvidas técnicas

---

Obrigado por contribuir para melhorar o MedVision Assist! 🚀
