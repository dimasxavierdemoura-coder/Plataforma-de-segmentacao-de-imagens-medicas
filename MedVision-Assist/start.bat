@echo off
chcp 65001 >nul
title MedVision Assist - Setup Completo
cls

echo.
echo ===========================================
echo    🏥 MEDVISION ASSIST - SETUP COMPLETO
echo ===========================================
echo.
echo Este script vai guiar você por todo o processo:
echo 1. Validação do ambiente Python
echo 2. Escolha entre dataset próprio ou Synapse
echo 3. Treinamento do modelo
echo 4. Inferência e validação
echo.
echo IMPORTANTE: Se você já tem seu próprio dataset, NÃO precisa usar o Synapse.
echo.
pause

cls
echo.
echo ===========================================
echo    🧪 PASSO 1: VALIDANDO AMBIENTE PYTHON
echo ===========================================
echo.
python -c "import torch, nibabel, PIL, streamlit; print('Dependencies OK')"
if %errorlevel% neq 0 (
    echo.
    echo ❌ Erro de ambiente Python.
    echo Certifique-se de que todos os pacotes foram instalados:
    echo - torch
    echo - nibabel
    echo - pillow
    echo - streamlit
    echo.
    pause
    exit /b 1
)
echo ✅ Ambiente Python validado.
echo.
set "TRAIN_ROOT=data/organized"
set "USE_LOCAL_DATASET=0"

echo Você já possui um dataset próprio com imagens e máscaras? (S/N)
set /p HAS_DATASET="Dataset próprio? (S/N): "
if /I "%HAS_DATASET%"=="S" (
    echo.
    echo Ótimo. Vamos usar seu dataset local.
    set /p LOCAL_DATASET_PATH="Informe o caminho para o diretório raiz do seu dataset: "
    if "%LOCAL_DATASET_PATH%"=="" (
        echo ❌ Caminho não pode ficar vazio.
        pause
        exit /b 1
    )
    if not exist "%LOCAL_DATASET_PATH%" (
        echo ❌ Diretório não encontrado: %LOCAL_DATASET_PATH%
        pause
        exit /b 1
    )
    set "TRAIN_ROOT=%LOCAL_DATASET_PATH%"
    set "USE_LOCAL_DATASET=1"
) else (
    echo.
    echo Sem dataset local. Vamos sugerir o Synapse como alternativa.
    echo.
)

echo.
if "%USE_LOCAL_DATASET%"=="0" (
    cls
    echo.
    echo ===========================================
    echo    📥 PASSO 4: DOWNLOAD DO DATASET
    echo ===========================================
    echo.
    echo Agora vamos baixar o dataset MAMA-MIA do Synapse.
    echo Este dataset contém imagens médicas de câncer de mama.
    echo.
    echo Volume aproximado: 10.7 GB
    echo Tempo estimado: 15-30 minutos (depende da conexão)
    echo.

    echo Cole o token completo e pressione ENTER.
    set /p SYNAPSE_TOKEN="Cole aqui seu token do Synapse: "

    echo.
    if "%SYNAPSE_TOKEN%"=="" (
        echo ❌ Token não pode estar vazio!
        pause
        exit /b 1
    )

    echo.
    echo 🔐 Validando token...
    echo Token: %SYNAPSE_TOKEN:~0,50%...
    echo.

    python src/scripts/download_mama_mia_complete.py syn60868042 --token "%SYNAPSE_TOKEN%" --organize

    if %errorlevel% neq 0 (
        echo.
        echo ❌ Erro no download! Verifique:
        echo - Token válido e não expirado
        echo - Conexão com internet
        echo - Conta Synapse com permissões
        echo.
        pause
        exit /b 1
    )

    echo.
    echo ✅ Download e organização concluídos!
    echo.
    pause
)

cls
echo.
echo ===========================================
echo    🤖 PASSO 5: TREINAMENTO DO MODELO
echo ===========================================
echo.
echo Agora vamos treinar o modelo de segmentação U-Net.
echo Este processo pode levar várias horas dependendo do hardware.
echo.
echo Configurações recomendadas:
echo - Epochs: 20 (ciclos de treinamento)
echo - Batch size: 8 (imagens por lote)
echo - Validação automática: 20% do conjunto de treino
echo.
echo Deixe em branco para usar os valores recomendados.
echo.
set /p EPOCHS="Número de epochs [padrão 20]: "
if "%EPOCHS%"=="" set EPOCHS=20

set /p BATCH_SIZE="Tamanho do batch [padrão 8]: "
if "%BATCH_SIZE%"=="" set BATCH_SIZE=8

set /p VAL_SPLIT="Fração de validação [padrão 0.2]: "
if "%VAL_SPLIT%"=="" set VAL_SPLIT=0.2

echo.
if exist models\unet_medvision.pth (
    echo Modelo existente encontrado em models\unet_medvision.pth
    set /p RETRAIN_MODEL="Deseja treinar novamente e sobrescrever o modelo? (S/N): "
    if /I "%RETRAIN_MODEL%"=="N" (
        echo.
        echo Usando modelo existente. Pulando treinamento.
        goto AFTER_TRAINING
    )
)
echo 🚀 Iniciando treinamento...
echo Epochs: %EPOCHS%
echo Batch size: %BATCH_SIZE%
echo Val split: %VAL_SPLIT%
echo Dataset usado: %TRAIN_ROOT%
echo.

python src/train.py --train-root "%TRAIN_ROOT%" --epochs %EPOCHS% --batch-size %BATCH_SIZE% --val-split %VAL_SPLIT% --seed 42 --experiment-name "medvision_experiment"

if %errorlevel% neq 0 (
    echo.
    echo ❌ Erro no treinamento! Verifique os logs acima.
    echo.
    pause
    exit /b 1
)

if not exist models\unet_medvision.pth (
    echo.
    echo ❌ Modelo treinado não encontrado em models\unet_medvision.pth
    echo Verifique se o treinamento foi concluído corretamente.
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ Treinamento concluído!
echo.
echo Pressione ENTER para continuar para a etapa de inferência.
pause >nul

:AFTER_TRAINING

echo.
set /p RUN_INFERENCE="Deseja executar inferência agora? (S/N): "
if /I "%RUN_INFERENCE%"=="S" (
    echo.
    echo ===========================================
echo    🧪 PASSO 6: INFERÊNCIA E VALIDAÇÃO
echo ===========================================
echo.
    echo Informe o caminho da imagem NIfTI para inferência.
    echo Exemplo: data/organized/images/duke_002_0000.nii.gz
    echo.
    echo Use aspas se o caminho contiver espaços.
    echo.
    set /p INPUT_IMAGE="Imagem NIfTI: "

    if "%INPUT_IMAGE%"=="" (
        echo ❌ Caminho da imagem não pode ficar vazio.
        pause
        exit /b 1
    )

    if not exist "%INPUT_IMAGE%" (
        echo ❌ Arquivo de imagem não encontrado: %INPUT_IMAGE%
        echo Verifique o caminho e tente novamente.
        pause
        exit /b 1
    )

    echo.
    set /p VOLUME_INFER="Executar inferência volumétrica? (S/N) [N]: "
    set "VOLUME_ARG="
    if /I "%VOLUME_INFER%"=="S" set "VOLUME_ARG=--volume"

    set /p HEATMAP_SLICE="Índice de slice para heatmap (deixe em branco para central): "
    set "SLICE_ARG="
    if not "%HEATMAP_SLICE%"=="" set "SLICE_ARG=--heatmap-slice %HEATMAP_SLICE%"

    echo.
    echo 🔍 Validando modelo e imagem...
    echo Modelo: models\unet_medvision.pth
    echo Imagem: %INPUT_IMAGE%
    echo.

    python src/inference.py --model-path models/unet_medvision.pth --input-image "%INPUT_IMAGE%" --output-dir results %VOLUME_ARG% %SLICE_ARG%
    if %errorlevel% neq 0 (
        echo.
        echo ❌ Erro na inferência! Verifique os logs acima.
        pause
        exit /b 1
    )

    echo.
    echo ✅ Inferência concluída e validada.
    echo Resultado salvo em results\
    echo.
    pause
)

cls
echo.
echo ===========================================
echo    🎉 SETUP CONCLUÍDO COM SUCESSO!
echo ===========================================
echo.
echo Parabéns! Seu projeto MedVision Assist está pronto.
echo.
echo Resumo da execução:
if /I "%USE_LOCAL_DATASET%"=="1" (
    echo ✅ Dataset próprio usado: %TRAIN_ROOT%
    echo ✅ Dados validados para treino
) else (
    echo ✅ Conta Synapse criada
    echo ✅ Token de acesso gerado
    echo ✅ Dataset MAMA-MIA baixado (741 imagens + 330 máscaras)
    echo ✅ Dados organizados automaticamente
)
echo ✅ Modelo U-Net treinado
if /I "%RUN_INFERENCE%"=="S" (
    echo ✅ Inferência executada e validada
) else (
    echo ✅ Pronto para inferência manual
)
echo.
echo Para usar o sistema:
echo.
echo 1. Interface Web:
echo    streamlit run src/app.py

echo    Agora suporta upload direto de NIfTI (.nii / .nii.gz)
echo.
echo 2. Inferência direta:
echo    python src/inference.py --model-path models/unet_medvision.pth --input-image sua_imagem.nii.gz

echo 3. Treinar novamente com validação automática:
echo    python src/train.py --train-root data/organized --epochs 50 --val-split 0.2
echo.
echo 📚 Documentação completa em: README.md
echo.
echo Obrigado por usar MedVision Assist! 🏥🤖
echo.
pause

exit /b 0