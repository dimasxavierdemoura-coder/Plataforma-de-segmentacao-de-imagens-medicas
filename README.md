# MedVision Assist

# 📊 Descritivo do Projeto MedVision Assist

O MedVision Assist é uma plataforma automatizada de visão computacional médica desenvolvida para auxiliar profissionais de saúde e pesquisadores na segmentação e análise de imagens médicas, com foco em aplicações como diagnóstico assistido por IA. O projeto foi implementado como um workflow completo no Windows, elevando-o a um padrão de qualidade profissional (inspirado em práticas de organizações como a NASA), incluindo automação, reprodutibilidade, testes e preparação para produção.

# 🚀 O Que o Projeto Faz?
O MedVision Assist automatiza o ciclo completo de desenvolvimento e uso de modelos de IA para segmentação de imagens médicas, especificamente voltado para dados volumétricos como ressonâncias magnéticas (MRI) ou tomografias computadorizadas (CT). Ele permite:

Download e organização automática de datasets: Baixa datasets públicos (como do Medical Segmentation Decathlon) via Synapse ou permite o uso de dados locais.
Treinamento de modelos: Treina um modelo U-Net para segmentar regiões de interesse (ex.: tumores ou órgãos) em imagens médicas, com controle de reprodutibilidade, métricas clínicas e relatórios.
Inferência e análise: Processa imagens individuais ou em lote, suportando formatos NIfTI e DICOM, gerando máscaras de segmentação, métricas clínicas (como Dice Score, IoU, sensibilidade) e relatórios detalhados.
Interface interativa: Uma aplicação web Streamlit para upload de imagens, visualização de volumes (slices), comparação de máscaras e exibição de métricas em tempo real.
Validação e testes: Inclui testes automatizados para garantir precisão e robustez, além de containerização Docker para implantação.
Relatórios e logs: Gera CSVs, JSONs e heatmaps para análise de desempenho, facilitando auditoria e melhorias.
O projeto transforma dados brutos em insights acionáveis, como estimativas de volume de lesões ou comparações de segmentações, ajudando em decisões clínicas ou pesquisas.

# 🛠️ Quais Tecnologias Utiliza?
O projeto é construído com tecnologias modernas e open-source para garantir eficiência, escalabilidade e acessibilidade:

Python 3.10+: Linguagem principal, com type hints para robustez.
PyTorch 2.0+: Framework de deep learning para treinamento e inferência do modelo U-Net.
Streamlit: Framework web para a interface interativa, permitindo uploads e visualizações sem necessidade de servidor complexo.
Nibabel e pydicom: Bibliotecas para manipulação de imagens médicas em formatos NIfTI (volumétricos) e DICOM (padrão clínico).
scikit-learn e NumPy: Para cálculos de métricas clínicas e operações numéricas.
Albumentations: Para augmentations de imagens durante o treinamento, melhorando a generalização do modelo.
Synapse client: Para download automatizado de datasets de repositórios médicos.
Docker: Containerização para ambientes reprodutíveis e implantação em produção.
pytest: Framework de testes automatizados para validar funções e inferência.
Outros: Bibliotecas como pandas para relatórios, matplotlib para visualizações e tqdm para barras de progresso.
O ambiente é configurado via start.bat para Windows, com virtual environments e dependências gerenciadas por requirements.txt.

# 🎯 Qual é o Foco do Projeto?
O foco principal é a segmentação precisa e automatizada de imagens médicas volumétricas, com ênfase em aplicações clínicas como detecção de tumores cerebrais ou análise de órgãos. Ele prioriza:

Precisão clínica: Uso de métricas como Dice Score e IoU para avaliar desempenho médico-realista.
Reprodutibilidade: Controle de seeds, logs de experimentos e relatórios para pesquisa científica.
Acessibilidade: Suporte a múltiplos formatos (NIfTI/DICOM), processamento em lote e interface amigável para não-especialistas em IA.
Qualidade profissional: Testes rigorosos, documentação completa e preparação para produção, visando uso em ambientes hospitalares ou de pesquisa.
Não é um modelo genérico de visão computacional, mas especializado em medicina, com validação contra datasets públicos para garantir confiabilidade.

# 👥 Qual a Usabilidade?
O projeto é altamente usável, projetado para profissionais com conhecimentos básicos em Python ou medicina, sem necessidade de expertise avançada em IA:

Automação total: O script start.bat guia o usuário desde o download de dados até o treinamento e inferência, com prompts para opções locais ou remotas.
Interface intuitiva: A app Streamlit permite upload de arquivos via drag-and-drop, visualização interativa de slices volumétricos e comparação de resultados em segundos.
Flexibilidade: Suporta processamento único ou em lote, com saídas em formatos padrão (JSON, CSV, imagens) para integração com ferramentas clínicas.
Robustez: Trata erros comuns (ex.: arquivos corrompidos), valida datasets e gera logs detalhados para depuração.
Escalabilidade: Pode ser executado localmente no Windows ou containerizado via Docker para servidores.
É ideal para prototipagem rápida em pesquisa ou uso clínico diário, com tempo de setup reduzido a minutos.

# 💼 O Que Pode Acrescentar no Dia-a-Dia do Profissional?
Para médicos, radiologistas e pesquisadores em saúde, o MedVision Assist acelera e aprimora o trabalho diário:

Diagnóstico mais rápido e preciso: Segmenta automaticamente regiões de interesse em exames, reduzindo tempo de análise manual (de horas para minutos) e minimizando erros humanos.
Suporte a decisões clínicas: Fornece métricas quantitativas (ex.: volume de tumor estimado) para planos de tratamento, melhorando prognósticos e reduzindo custos hospitalares.
Pesquisa eficiente: Facilita experimentos reprodutíveis com datasets públicos, permitindo comparações de modelos e publicações científicas mais rápidas.
Integração com fluxos clínicos: Compatível com DICOM, padrão em hospitais, e gera relatórios exportáveis para sistemas EHR (Electronic Health Records).
Redução de carga de trabalho: Automatiza tarefas repetitivas, liberando tempo para foco em pacientes ou inovação, enquanto mantém alta precisão (métricas clínicas validadas).
Benefícios profissionais: Aumenta produtividade, reduz burnout e eleva a qualidade do cuidado, potencialmente impactando milhares de pacientes por ano em ambientes de alta demanda.
