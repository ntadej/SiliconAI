# A script to setup PDM for usage and installs it if needed

# Bash location
SETUP_LOCATION=${BASH_SOURCE[0]}
# Zsh fallback
if [[ -z ${BASH_SOURCE[0]+x} ]]; then
    SETUP_LOCATION=${(%):-%N}
fi

SILICONAI_PATH=$(cd "$(dirname "${SETUP_LOCATION}")" && cd .. && pwd)
SILICONAI_PARENT_PATH=$(cd "$(dirname "${SETUP_LOCATION}")" && cd ../.. && pwd)

PDM_HOME="${SILICONAI_PARENT_PATH}/pdm"
export PDM_HOME

echo "SiliconAI path: ${SILICONAI_PATH}"
echo "SiliconAI parent path: ${SILICONAI_PARENT_PATH}"
echo "PDM path: ${PDM_HOME}"

# Check if PDM is installed
if [[ -d "${PDM_HOME}" ]]; then
    echo "  PDM is already installed"
else
    echo "  PDM is not installed. Installing PDM..."
    curl -sSL https://pdm-project.org/install-pdm.py | python3 -
fi

# Add PDM to PATH
if [[ ":${PATH}:" != *":${PDM_HOME}/bin:"* ]]; then
    export PATH="${PATH}:${PDM_HOME}/bin"
    echo "PATH updated to include PDM: ${PDM_HOME}/bin"
fi
