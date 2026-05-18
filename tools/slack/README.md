# lynx Cluster Monitor — Slack 通知ツール

lynx HPC クラスターの状態を定期的に Slack へ通知するスクリプト。

## 通知内容

| セクション | 詳細 |
|---|---|
| 📊 Node Health | パーティション別ノード稼働状況。DRAIN/DOWN ノードを個別列挙 |
| ⚡ Job Queue | 全ユーザーの実行中・待機中ジョブ一覧 |
| 💾 Storage | `/data3` 使用量・空き容量（70%以上で🟡、90%以上で🔴） |
| 🔥 GPU Status | アクティブノード（最大6台）の温度・メモリ・利用率 |

メッセージ左端の色：🟢 正常 / 🟡 軽度異常 / 🔴 重大異常

## 定期実行（cron）

lynx 上に cron が設定済み。毎日 **0:00 / 6:00 / 12:00 / 18:00 JST** に自動実行。

```
# crontab -l で確認
0 0,6,12,18 * * * /usr/bin/python3 /data3/kusumoto/elastodynamics-fem/tools/slack/monitor_slack.py >> /data3/kusumoto/elastodynamics-fem/cpp_log/monitor_slack.log 2>&1
```

## 手動実行

```bash
# lynx 上で
export PATH=$PATH:/opt/slurm/22.05.2/bin
python3 /data3/kusumoto/elastodynamics-fem/tools/slack/monitor_slack.py
```

WSL から SSH 越しに実行する場合:

```bash
cp /mnt/c/.ssh/id_rsa /tmp/lynx_id_rsa && chmod 600 /tmp/lynx_id_rsa
ssh -i /tmp/lynx_id_rsa kusumoto@lynx.eri.u-tokyo.ac.jp \
  "export PATH=\$PATH:/opt/slurm/22.05.2/bin && \
   python3 /data3/kusumoto/elastodynamics-fem/tools/slack/monitor_slack.py"
```

## ログ

実行ログは `cpp_log/monitor_slack.log` に追記される。

```bash
# lynx 上で確認
tail -f /data3/kusumoto/elastodynamics-fem/cpp_log/monitor_slack.log
```

## スクリプト更新手順

1. ローカルで `tools/slack/monitor_slack.py` を編集
2. lynx へ転送（WSL から）:

```bash
KEY="-i /tmp/lynx_id_rsa"
rsync -av -e "ssh $KEY" \
  tools/slack/monitor_slack.py \
  kusumoto@lynx.eri.u-tokyo.ac.jp:/data3/kusumoto/elastodynamics-fem/tools/slack/
```

3. 手動実行でテスト（上記参照）
4. 問題なければ次のcron実行から自動的に新バージョンが使われる

## cron の再登録（サーバーメンテ後など）

```bash
# lynx にSSH接続後
SCRIPT="/data3/kusumoto/elastodynamics-fem/tools/slack/monitor_slack.py"
LOG="/data3/kusumoto/elastodynamics-fem/cpp_log/monitor_slack.log"
( crontab -l 2>/dev/null | grep -v "monitor_slack.py"; \
  echo "0 0,6,12,18 * * * /usr/bin/python3 $SCRIPT >> $LOG 2>&1" ) | crontab -
crontab -l   # 確認
```

## 主要パラメータ（monitor_slack.py 内）

| 変数 | 説明 |
|---|---|
| `WEBHOOK_URL` | Slack Incoming Webhook URL |
| `PARTITION_ORDER` | 表示するパーティションの順序 |
| `GPU_PARTITIONS` | GPU監視対象のパーティション |
| `active[:6]` の `6` | GPU監視を行うノードの最大数 |

## トラブルシューティング

**Slack に届かない**
- ログを確認: `tail cpp_log/monitor_slack.log`
- Webhook URL が有効か確認（Slack App 設定 → Incoming Webhooks）
- lynx からインターネット疎通確認: `curl -s https://hooks.slack.com`

**GPU Status が表示されない**
- アクティブノードがなければ正常（セクション自体が省略される）
- アクティブなのに表示されない場合: ログインノードから計算ノードへの SSH を確認
  ```bash
  ssh -o BatchMode=yes -o UserKnownHostsFile=/dev/null lynx01 nvidia-smi
  ```

**ノード状態が正確でない**
- `sinfo` のパス確認: `export PATH=$PATH:/opt/slurm/22.05.2/bin && sinfo`
