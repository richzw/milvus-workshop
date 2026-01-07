# 在 Docker 中运行 Milvus (Linux)

本页说明如何在 Docker 中启动 Milvus 实例。

## 前提条件

- [安装 Docker](https://docs.docker.com/get-docker/)。
- 安装前[请检查硬件和软件要求](https://milvus.io/docs/zh/prerequisite-docker.md)。

## 在 Docker 中安装 Milvus

Milvus 提供了一个安装脚本，可将其安装为 docker 容器。该脚本可在[Milvus 存储库中](https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh)找到。要在 Docker 中安装 Milvus，只需运行

```shell
# Download the installation script
$ curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

# Start the Docker container
$ bash standalone_embed.sh start
```

如果你想在独立部署模式下使用[备份](https://milvus.io/docs/milvus_backup_overview.md)，建议使用[Docker Compose](https://milvus.io/docs/install_standalone-docker-compose.md)部署方法。

如果在拉取镜像时遇到任何问题，请通过[community@zilliz.com](mailto:community@zilliz.com)联系我们并提供有关问题的详细信息，我们将为您提供必要的支持。

运行安装脚本后

- 一个名为 Milvus 的 docker 容器已在**19530** 端口启动。
- 嵌入式 etcd 与 Milvus 安装在同一个容器中，服务端口为**2379**。它的配置文件被映射到当前文件夹中的**embedEtcd.yaml。**
- 要更改 Milvus 的默认配置，请将您的设置添加到当前文件夹中的**user.yaml**文件，然后重新启动服务。
- Milvus 数据卷被映射到当前文件夹中的**volumes/milvus**。

你可以访问 Milvus WebUI，网址是`http://127.0.0.1:9091/webui/` ，了解有关 Milvus 实例的更多信息。有关详细信息，请参阅[Milvus WebUI](https://milvus.io/docs/zh/milvus-webui.md)。

## 停止和删除 Milvus

停止和删除此容器的方法如下

```shell
# Stop Milvus
$ bash standalone_embed.sh stop

# Delete Milvus data
$ bash standalone_embed.sh delete
```

你可以按以下步骤升级最新版本的 Milvus

```shell
# upgrade Milvus
$ bash standalone_embed.sh upgrade
```

## 下一步

在 Docker 中安装 Milvus 后，你可以

- 查看[快速入门](https://milvus.io/docs/zh/quickstart.md)，了解 Milvus 的功能。
- 了解 Milvus 的基本操作：
  - [管理数据库](https://milvus.io/docs/zh/manage_databases.md)
  - [管理 Collections](https://milvus.io/docs/zh/manage-collections.md)
  - [管理分区](https://milvus.io/docs/zh/manage-partitions.md)
  - [插入、倒置和删除](https://milvus.io/docs/zh/insert-update-delete.md)
  - [单向量搜索](https://milvus.io/docs/zh/single-vector-search.md)
  - [混合搜索](https://milvus.io/docs/zh/multi-vector-search.md)
- [使用 Helm 图表升级 Milvus](https://milvus.io/docs/zh/upgrade_milvus_cluster-helm.md)。
- [扩展你的 Milvus 集群](https://milvus.io/docs/zh/scaleout.md)。
- 在云上部署你的 Milvu 集群：
  - [亚马逊 EKS](https://milvus.io/docs/zh/eks.md)
  - [谷歌云](https://milvus.io/docs/zh/gcp.md)
  - [微软 Azure](https://milvus.io/docs/zh/azure.md)
- 探索[Milvus WebUI](https://milvus.io/docs/zh/milvus-webui.md)，一个用于 Milvus 可观察性和管理的直观 Web 界面。
- 探索[Milvus 备份](https://milvus.io/docs/zh/milvus_backup_overview.md)，一个用于 Milvus 数据备份的开源工具。
- 探索[Birdwatcher](https://milvus.io/docs/zh/birdwatcher_overview.md)，用于调试 Milvus 和动态配置更新的开源工具。
- 探索[Attu](https://github.com/zilliztech/attu)，一个用于直观管理 Milvus 的开源图形用户界面工具。
- [使用 Prometheus 监控 Milvus](https://milvus.io/docs/zh/monitor.md)。