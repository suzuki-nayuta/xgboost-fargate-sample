# 『Fargateタスクで機械学習モデルを作成・実行したのでポイントを整理してみた』のサンプルコード

## コードについて
### CloudFormationテンプレート
以下のテンプレートを`cloudformation`ディレクトリに入れています。
- 学習用Fargate関連リソースのテンプレート
- 推論用Fargate関連リソースのテンプレート
- VPCエンドポイントのテンプレート

VPCは既に構築されている想定です。

## 学習・推論用イメージ
`docker-asset-train`および`docker-asset-inference`ディレクトリに各種Pythonスクリプトや`dockerfile`などを格納しています。

### データ作成例
`data_create`ディレクトリに作成に使用したJupyterノートブックの例を格納しています。

## 参考資料
以下の資料を参考にしつつ、コードを作成しました。
- [Amazon ECS resource type reference \- AWS CloudFormation](https://docs.aws.amazon.com/ja_jp/AWSCloudFormation/latest/UserGuide/AWS_ECS.html)
- [\[AWS\]ECS環境を作成するCloudFormationのテンプレート \| 個人利用で始めるAWS学習記](https://noname.work/3105.html)
- [CloudformationでFargateを構築する \| DevelopersIO](https://dev.classmethod.jp/articles/cloudformation-fargate/)