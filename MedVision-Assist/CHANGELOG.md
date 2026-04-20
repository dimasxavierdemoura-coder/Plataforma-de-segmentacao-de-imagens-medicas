# Changelog

Todas as mudanças notáveis neste projeto são documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
e este projeto adere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-20

### Added
- ✅ Segmentação U-Net para imagens médicas
- ✅ Suporte a formatos NIfTI (.nii, .nii.gz)
- ✅ Suporte a formatos DICOM (.dcm)
- ✅ Métricas clínicas: Dice, IoU, Sensibilidade, Especificidade, F1-Score
- ✅ Interface Streamlit interativa com visualização 2D
- ✅ Processamento em lote (batch processing)
- ✅ Treinamento reprodutível com seed control
- ✅ Geração de relatórios JSON estruturados
- ✅ Logging automático de experimentos
- ✅ Testes unitários com pytest (6/6 passando)
- ✅ Dockerização para produção
- ✅ Suporte a comparação de máscaras (predita vs. referência)
- ✅ Visualização de slices para volumes NIfTI
- ✅ Exportação de resultados e mascaras

### Infrastructure
- Dockerfile pronto para produção
- .gitignore e .dockerignore configurados
- Requirements.txt com todas as dependências
- Estrutura de testes com cobertura
- Script de setup (start.bat) para Windows
- Documentação completa (README, Dataset Setup, Contributing)

### Documentation
- README.md com instruções de quick start
- DATASET_SETUP.md com guias de datasets públicos
- CONTRIBUTING.md com padrões de código
- CHANGELOG.md (este arquivo)

## Roadmap para v2.0

### Planned
- [ ] Integração com Azure AI Foundry
- [ ] Visualização 3D interativa (Plotly/VTK)
- [ ] Autenticação OAuth2 (Azure AD)
- [ ] Dashboard de métricas em tempo real
- [ ] Suporte a multi-GPU (DistributedDataParallel)
- [ ] API FastAPI para integração
- [ ] CI/CD com GitHub Actions
- [ ] Otimização de performance (ONNX Runtime, TensorRT)
- [ ] Detecção de anomalias não supervisionada
- [ ] Aprendizado ativo com feedback de usuário

### Under Consideration
- [ ] Suporte a segmentação multi-classe
- [ ] Ensemble de modelos
- [ ] Fine-tuning com transfer learning
- [ ] Integração com ferramentas DICOM (PACS)
- [ ] Suporte a dados em tempo real de scanners
- [ ] Conformidade HIPAA/GDPR

---

**Nota**: Este projeto mantém um histórico completo de desenvolvimento. 
Para mais informações, consulte [README.md](README.md) e [CONTRIBUTING.md](CONTRIBUTING.md).
