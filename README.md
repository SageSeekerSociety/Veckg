# Veckg：基于 DeepSeek 和词向量的知识图谱自动化构建工具

Text -> Knowledge Graph -> Embeddings

## 环境配置

考虑到这个项目的很多使用者可能来自各个不同专业，不一定熟悉 Python 环境配置的一些基础知识，所以我在这个文档里写的较为细致。如果你发现自己对此已经很了解，可以直接跳过“环境配置”这一部分。

### 1. 创建 Python 虚拟环境

Python虚拟环境是一个隔离的工作空间，可以让你在同一台电脑上同时使用不同版本的Python和不同的库。它就像是给每个项目准备了一个独立的小“沙盒”，避免了不同项目之间的库冲突。例如，项目A需要用某个版本的库，项目B需要另一个版本，虚拟环境可以让它们各自运行，而不会互相影响。

简而言之，虚拟环境可以帮助你更好地管理项目依赖，并且保持环境的整洁，不会因为不同的项目需求导致混乱。

在PyCharm或VSCode里，创建并使用Python虚拟环境其实非常简单。你只需要打开编辑器自带的终端（就是在编辑器下方的一个命令行窗口），然后通过一些简单的指令来完成。这样，你就能为每个项目创建一个独立的工作环境，避免不同项目之间的库或版本冲突。

1. 打开PyCharm后，进入你的项目。你会看到编辑器下方有一个“Terminal”（终端）选项，点击它就能打开一个命令行窗口。
2. 在终端里，输入这条命令来创建一个虚拟环境：

   ```bash
   python -m venv myenv
   ```

   这会在你的项目文件夹里创建一个名为`myenv`的虚拟环境，`myenv`其实就是你项目的“独立空间”。
3. 创建完成后，你需要激活这个虚拟环境。在终端里输入以下命令：
   - 如果你使用的是Windows电脑，输入：

     ```bash
     myenv\Scripts\activate
     ```

   - 如果你使用的是Mac或者Linux电脑，输入：

     ```bash
     source myenv/bin/activate
     ```

   激活后，你会看到终端的前面有个`(myenv)`，表示你现在就在虚拟环境中工作了。

4. **让PyCharm自动使用虚拟环境：**  
   为了让PyCharm在每次启动时自动使用这个虚拟环境，你可以设置PyCharm的Python解释器。这样，你不需要每次手动激活虚拟环境。
   - 在PyCharm顶部菜单栏找到“File”（文件），然后点击“Settings”（设置）。
   - 在设置窗口中，找到“Project: [你的项目名]”下的“Python Interpreter”（Python 解释器）。
   - 点击右侧的齿轮图标，选择“Add”（添加）。
   - 选择“Existing environment”（现有环境），然后找到并选择你刚才创建的虚拟环境中的`python.exe`（在`myenv\Scripts`文件夹中）。
   - 点击“OK”保存设置，PyCharm就会自动使用你创建的虚拟环境。

这样设置完之后，每次你在PyCharm中运行代码时，都会自动使用你为项目配置的虚拟环境，避免了不同项目之间的冲突。

在VSCode中操作类似，只不过在VSCode里，你需要通过右下角的Python版本选择框来选择虚拟环境，并且也可以在内置终端中创建和激活虚拟环境。

### 2. 安装相关包

在你创建了虚拟环境并激活之后，你可能需要安装一些依赖的Python包，这些包是你项目运行所必须的。通常，这些包会列在一个名为`requirements.txt`的文件里，这个文件就像是一个清单，列出了项目所需的所有包和它们的版本。

1. **打开终端**：在PyCharm或VSCode中，打开你刚才用来创建虚拟环境的终端窗口。

2. **确认虚拟环境已激活**：在终端中，确认你看到的提示符前面有`(myenv)`，这表示你已经进入了虚拟环境。如果没有，记得先激活它。

3. **安装依赖包**：在终端中，输入以下命令来安装`requirements.txt`文件里列出的所有包：

   ```bash
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
   ```

   这个命令会做两件事：
   - **`-i https://pypi.tuna.tsinghua.edu.cn/simple`**：这个部分指定了使用清华大学的镜像源来安装包，通常比默认的源要更快·。
   - **`-r requirements.txt`**：这个部分表示从`requirements.txt`文件里读取依赖包的清单，并根据这个清单自动安装所有的包。

4. **等待安装完成**：运行这个命令后，`pip`会自动从指定的源下载并安装文件里列出的所有包。你只需要等几分钟，直到所有包安装完成。

安装完成后，你就可以在项目中使用这些依赖的包了。如果有包安装失败，通常会有错误提示，你可以根据提示去解决问题·。

这样，你就完成了通过`pip`安装项目所需的所有依赖包，并且这些包只会影响到当前的虚拟环境，不会干扰到系统中的其他项目。

### 3. 配置 DeepSeek API Key

#### 什么是DeepSeek API Key？

可以把 **DeepSeek API Key** 想象成你访问一个高端俱乐部的会员卡。这个 API 密钥就像是一个特殊的凭证，只有拥有这个凭证的人才能进入俱乐部并享受它的服务。你需要这个 API 密钥来证明你是被允许使用 DeepSeek 平台的服务的。

API（应用程序编程接口）就像是俱乐部的接待员，而 API 密钥则是你给接待员看的一张会员卡。通过这张卡，接待员知道你有资格使用俱乐部里的设施（在这里就是 DeepSeek 平台的功能）。如果没有正确的 API 密钥，接待员就不允许你进入。

在你的项目中，API 密钥就像一个“钥匙”，让你的程序能够“开门”访问 DeepSeek 平台，获取它提供的数据或功能。

除此之外，DeepSeek API Key还涉及到计费和额度管理。可以想象，俱乐部不仅会给你一个会员卡，还会记录你使用的服务次数。如果你使用得太多，可能会超过你的“额度”或“会员福利”，这时你就需要根据使用量来支付费用。所以，API Key不仅是你进入平台的“钥匙”，还是记录你使用量的凭证。DeepSeek平台会根据你使用API的次数、请求的数据量来进行计费，而你的API密钥就会与这些使用记录关联起来。

#### 那么具体来说如何配置呢？

1. **复制并重命名文件**：在项目文件夹里，你会找到一个名为`sample.env`的文件，它像是一个模板。你需要做的是复制这个模板文件，然后将它重命名为`.env`。就像是你拿到一张空白的会员卡，需要写上你的名字，才能使用。

2. **编辑`.env`文件**：打开`.env`文件，你会看到类似这样的内容：

   ```text
   DEEPSEEK_API_KEY=you_api_key
   ```

   这就像是一张空的会员卡，上面标注了你需要填写自己的密钥。将`you_api_key`替换成你从DeepSeek平台获得的实际API密钥。假设你获得的密钥是`123456abcdef`，那你需要将这一行改成：

   ```text
   DEEPSEEK_API_KEY=123456abcdef
   ```

3. **保存文件**：修改完成后，保存这个文件。现在，你的`.env`文件就像是一张带有你名字的会员卡，已经准备好帮助你访问DeepSeek平台了。

#### 注意事项

- `.env`文件就像是你个人的秘密会员卡，不应该公开分享到物联网。他人使用你的 DeepSeek API Key 访问 DeepSeek 会消耗你在 DeepSeek 充值的余额。
- 配置好`.env`文件后，你的项目就能通过读取这个文件，自动获取API密钥，就像你的程序通过会员卡“进场”使用DeepSeek平台提供的服务。

通过这种方式，你的程序就像是一个合法的会员，可以安全、方便地使用DeepSeek平台的各种功能，而不会泄露你的密钥信息。

### 4. 下载模型

很好！到这一步你已经非常接近成功安装本工具了。请您观察一下这个 README.md 文档和 main.py 等代码文件所在的文件夹中，是否有一个名叫 model 的文件夹。**如果已经有这个文件夹了则说明模型已经下载好了，无需重新下载，可以跳过本步骤。**

如果没有 model 文件夹，请通过下面的命令运行 download_model.py

```bash
python download_model.py
```

待程序运行完成后，模型便下载成功了！
