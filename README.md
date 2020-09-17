---
page_type: sample
languages:
- python
products:
- azure-machine-learning-service
- azure-iot-edge
- azure-devops
---

# Azure DevOps Pipeline with ONNX Runtime

This sample will setup a pipeline to train, package and deploy Machine Learning models in IoT Edge Devices. There are three phases in this pipeline. (1) __Training__ the Tiny Yolo v3 model in Azure Machine Learning and converting it to ONNX. (2) __Packaging Pipeline__ to create the CI/CD steps to create docker image for the NVIDIA Jetson device with the application code and the ONNX model. (3) __Deploying__ the docker images on the target device using Azure IoT Edge. All these steps are automated in a DevOps pipeline using Azure DevOps.

Specifically, we will cover:
* How to set up a NVIDIA Jetson Nano as a Linux self-hosted DevOps agent, for building our Edge solution.
* How to trigger a release pipeline, when a newly trained model is registered in the AzureML model registry.

### Acknowledgements

The Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K). This sample reuses the recipes from the [qqwweee/keras-yolo3] (https://github.com/qqwweee/keras-yolo3) repo to train a keras-yolo3 model on the VOC Pascal dataset, using [AzureML](https://azure.microsoft.com/en-us/services/machine-learning/). If you are familiar with the original repository, you may want to jump right to section `Train On AzureML` below.

### A few things before you get started:

* __Setup you Azure account__: An Azure Account Subscription (with pre-paid credits or billing through existing payment channels) is required for this sample. Create the account in Azure portal using [this tutorial](https://azure.microsoft.com/en-us/free/). Your subscription must have pre-paid credits or bill through existing payment channels. (If you make an account for the first time, you can get 12 months free and $200 in credits to start with.)

* __Devices__ needed for this sample needs atleast _two_ NVIDIA Jetson devices. Read more about the NVIDIA Jetson Developer Kit [here](https://www.nvidia.com/en-us/autonomous-machines/jetson-store/).

    We will use one of the Jetson devices as the Azure DevOps self-host agent to run the jobs in the DevOps pipeline. This is the _Dev machine_.
    The other Jetson device(s) will be used to deploy the IoT application containers. We will refer to these devices as _Test Device(s)_.

    > Note: If you are ordering these devices, we recommend you get power adapters (rather than relying on USB as a power source) and a wireless antenna (unless you are fine with using ethernet).

* Before you try to create a DevOps release pipeline, we recommend that you familiarize yourself with [this](https://github.com/wmpauli/onnxruntime-iot-edge/blob/master/README-ONNXRUNTIME-arm64.md) easy to use getting started sample to deploy a ML model manually to an IoT Edge device like the Jetson device.

## <a name="S1"></a>1. Train On AzureML

In this step we will use the Tiny Yolo weight from the original release by Darknet. The training recipe is reused from We will first convert the weights to Keras to training with TensorFlow backend. This model is then training with the VOC dataset in AML. The trained is converted to ONNX to enable us to deploy in different execution environments.

[Create your Azure Machine Learning Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace). (_You can skip this step if you already have a workspace setup._)

[Setup the Jupyter Notebook Environment in Azure Machine Learning Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-run-jupyter-notebooks) to run your Jupyter Notebooks directly in your workspace in AML studio.

[Clone this repo to your AML Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-run-jupyter-notebooks#terminal) to run this training notebook.
You can use regular `git clone --recursive https://github.com/Azure-Samples/AzureDevOps-onnxrutime-jetson` CLI commands from the Notebook Terminal in AML to clone this repository into a desired folder in your workspace.

__Get Started to Train__: Open the notebook `Training-keras-yolo3-AML.ipynb` and start executing the cells to train the Tiny Yolo model. 

## <a name="S2"></a>2. Packaging Pipeline

In this step we will create the pipeline of steps to build the docker images for the Jetson devices. We will use [Azure DevOps](https://azure.microsoft.com/en-us/solutions/devops/) to create this pipeline.

#### Create a DevOps project

Go to [https://dev.azure.com/](https://dev.azure.com/) and create a new organization and project.

#### Setup Dev machine

The _Dev machine_ is setup to run the jobs for the CI/CD pipeline. Since the test device is a ubuntu/ARM64 platform we will need to build the ARM64 docker images on the host platform with same HW configuration. Another approach is to setup a docker cross-build environment in Azure which is beyond the scope of this tutorial and not fully validated for ARM64 configuration.

##### Azure DevOps Agent

Install the self-hosted Azure DevOps agent. Follow the instructions on this page: https://docs.microsoft.com/en-us/azure/devops/pipelines/agents/v2-linux?view=azure-devops.

##### Install Azure IoT Edge Dev Tool

The IoT Edge Dev Tool greatly simplifies Azure IoT Edge development down to simple commands driven by environment variables. We recommend that you install the tool manually on the ARM64: https://github.com/Azure/iotedgedev/wiki/manual-dev-machine-setup

##### Install core AzureML SDK for Python

We will use the AzureML SDK for Python to download the model to the DevOps agent.

You are welcome to just install the SDK system wide. Alternatively, you might want to install it inside a Conda environment - for easier housekeeping.

Because there is no official release of Anaconda/Miniconda for ARM64 devices, we recommend that you use [Archiconda](https://github.com/Archiconda/build-tools/releases).

Then, you can install the SDK like so:

```
conda create -n onnx python=3.7
conda activate onnx
pip install -U pip
pip install azureml-core
```

#### Config for AzureML workspace

> You can skip this step if you already have the configuration details for the AzureML workspace.

Note the configuration details of your AML Workspace: `config.json`

```
    {
        "subscription_id": "subscription_id",
        "resource_group": "resource_group",
        "workspace_name": "workspace_name",
        "workspace_region": "workspace_region",
        "service_principal_id": "service_principal_id",
        "service_principal_password": "service_principal_password",
        "tenant_id": "tenant_id"
    }
```

#### Create Service Principal

Service Principal enables non-interactive authentication for any specific user login. This is useful for setting up a machine learning workflow as an automated process.

> Note that you must have administrator privileges over the Azure subscription to complete these steps.

```
Follow the instructions of section "Service Principal Authentication" in [this notebook](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb). 

> We recommend to scope the Service Principal to the Resource Group.

**Note:** Add `service_principal_id`, `service_principal_password`, and `tenant_id` to the `config.json` file above. You can then upload the `config.json` file to the secure file libary of your DevOps project. Make sure to enable all pipelines to have access to the secure file.
```

Add config.json to library of secure files in the Azure DevOps project. Select on the rocket icon on the left, then the library. In your library go to *secure files* and *+ Secure File*. Upload your file and make sure that you allow all pipelines to use it.
<p align="center"><img width="100%" src="media/ado_lib.png" alt="Library in Azure DevOps project"/></p>


## Add Service Connections to your DevOps project

Next we configure your project such that the release pipeline has access to your fork of our github repo, to your AzureML Workspace, and to your Azure Container Registry (for Docker images).

Go to the settings of your project, `Service Connections` and click on `New Service Connection`.

- Create one Service Connection of type `GitHub`.
- Create one of type `Azure Resource Manager`, using the Service Principal Connection credentials from above.

## Install MLOps extension for Azure DevOps

You can install the MLOps extension from here: [https://marketplace.visualstudio.com/items?itemName=ms-air-aiagility.vss-services-azureml](https://marketplace.visualstudio.com/items?itemName=ms-air-aiagility.vss-services-azureml).

# Create Release Pipeline

Now we can build the Release pipeline. The final pipeline should look like this:

![schematic pipeline](./media/pipeline.png)

## Connect Artifacts

The pipeline is connected to two `Artifacts`, your fork of our GitHub repository and our model in the AzureML model registry.  You can add these by clicking the `+ Add` button, next to `Artifacts`.


If the pipeline is triggered, it will execute the tasks in `Stage 1`:

![tasks of stage 1](./media/stage_1.png)

Let's go through the steps indivually

### Download Secure file

![01_download_secure_file.png](./media/01_download_secure_file.png)

We called our file `wopauli_onnx_config.json`. Feel free to give it a different name. It helps to add some kind of identifier, in case you have other release pipelines that work with other AzureML Workspaces or Service Principals.

### Copy Secure file

![02_copy_secure_file.png](./media/02_copy_secure_file.png)

We copy the file from the Agent.TempDirectory into the aml folder below the root of the cloned code repository (`cp $(Agent.TempDirectory)/wopauli_onnx_config.json ./_wmpauli_onnxruntime-iot-edge/aml/config.json`).

> `Agent.TempDirectory` is a predefined variable. Check out what other predefined variables exist: [https://docs.microsoft.com/en-us/azure/devops/pipelines/build/variables](https://docs.microsoft.com/en-us/azure/devops/pipelines/build/variables)

### Download Model from AzureML Model Registry

![03_python_script.png](./media/03_python_script.png)

We use the AzureML SDK for Python to download the latest model from the Model Registry (`$(System.DefaultWorkingDirectory)/_wmpauli_onnxruntime-iot-edge/aml/download_model.py`)

### Build Modules

![04_build_modules.png](./media/04_build_modules.png)

We build the modules (docker images) of our solution.  Make sure you point it to the correct `deployment.template.json` file, and pick the correct `Default platform`, and `Action`.

### Push Modules

![05_push_modules.png](./media/05_push_modules.png)

The next step is to push the modules to the container registry.

You can use the Azure Container Registry that was create along your Workspace above. As `Azure Subscription`, pick the Service connection you created above to connect to your workspace.

### Deploy to Edge Device

![06_deploy.png](./media/06_deploy.png)

The last step is deploy the modules to the Edge device.


# Test

Now you can run `aml/model_registration.py` again. This should trigger a run of this release pipeline.  

*Note*: Make sure you clicked on the lightning Icon (continuous deployment trigger) on the `Artifact` `_TinyYOLO`, to make sure that the release pipeline is triggered when you register a new model in the Model Registry.

## <a name="S3"></a>3. Deployment




==============================
## Contribution

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.
 
When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

# Legal
 
## Code of conduct
 
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
 
## Reporting security issues
Security issues and bugs should be reported privately, via email, to the
Microsoft Security Response Center (MSRC) at [secure@microsoft.com](mailto:secure@microsoft.com).
You should receive a response within 24 hours. If for some reason you do not, please follow up via
email to ensure we received your original message. Further information, including the
[MSRC PGP](https://technet.microsoft.com/en-us/security/dn606155) key, can be found in the
[Security TechCenter](https://technet.microsoft.com/en-us/security/default).
 
## License
Copyright (c) Microsoft Corporation. All rights reserved.
 
MIT License
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
