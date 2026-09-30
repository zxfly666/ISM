"""Generate navigation/provenance tables from the public archive, not new science."""
import hashlib
import json
import os
import re
from pathlib import Path
from catalog import ROOT, C

OUT=ROOT/'experiments'
def j(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,t):p.write_text(t,encoding='utf-8')
def relative(p,start):return Path(os.path.relpath(p,start)).as_posix()

def main():
    for c in C:
        base=OUT/c['slug']; rec=j(base/'provenance.json')['files']
        lines=['# 证据与复现入口','',
          '本页由公开来源清单自动生成；科学数值未重新计算。文件名里的原始代号用于身份匹配，公开问题名见README。',
          '', '| 类型 | 入口 |','|---|---|',
          '| 设计与参数 | [协议及配置](protocol/)；下表中的run/frozen/effective protocol为实际记录 |',
          '| 原始/公开版本SHA | [provenance.json](provenance.json) |',
          '| 大文件/未公开材料 | [withheld-manifest.json](withheld-manifest.json)；不可把本地路径当下载地址 |',
          '| 本地备份核验（非公开托管） | [backup-verification.json](backup-verification.json) |','',
          '## 原科学代码','']
        lines += ['- ['+Path(p).name+']('+relative(ROOT/p,base)+')' for p in c['code']]
        lines += ['', '共享依赖及冻结身份：[全源码清单]('+relative(OUT/'source-manifest.json',base)+')。',
                  '', '## 机器结果、正式图与数据','',
                  '| 公开文件 | 字节 | SHA-256（完整值见provenance） |','|---|---:|---|']
        for r in rec:
            p=OUT/r['published']; rel=relative(p,base)
            lines.append('| ['+rel+']('+rel+') | '+str(r['bytes'])+' | `'+r['sha256'][:16]+'…` |')
        lines+=['','## 图的来源','',
                '此目录复用正式PNG/PDF；原统计JSON/CSV、NPZ和plot_data在上表。图的原文件SHA与输入源路径在provenance中。',
                '原分析代码中的figure/plot函数定义绘图映射。精确局部两轮若没有正式图，则用完整结果表与流程图，不拿人工fixture补作结果图。',
                '未公开的大数组仍可能是全图重建的输入；公开图不代表所有模型/父场已可下载。','']
        write(base/'EVIDENCE.md','\n'.join(lines))
        readme=base/'README.md';t=readme.read_text(encoding='utf-8')
        marker='\n## 证据、参数与可用性\n'
        t=t.split(marker)[0]
        t+=marker+'\n[完整机器证据与协议索引](EVIDENCE.md) · [逐文件来源/编辑SHA](provenance.json) · [科学路径及未公开材料](withheld-manifest.json) · [本地核验回执](backup-verification.json)。\n\n'
        t+='公开仓库不包含所有checkpoint、MC父场或bulk generation；从权重重跑与核对统计是不同复现层级。请先读[数据可用性]('+relative(OUT/'DATA_AVAILABILITY.md',base)+')和[复现说明]('+relative(OUT/'REPRODUCING.md',base)+')。\n\n'
        t+='[返回研究证据地图]('+relative(OUT/'README.md',base)+')。\n'
        write(readme,t)
    # Public reading edition: reproducible assembly of the new scientific summaries,
    # never the private transcript sections in the original report.
    original=ROOT/'output/pdf/临界Ising全阶段实验总报告_20260930.md'
    origsha=hashlib.sha256(original.read_bytes()).hexdigest()
    report=['# 临界Ising近期研究：公共综合报告','',
            '2026-09-30公共科研归档版。完整研究地图、独立实验与冻结诊断的区别、所有主失败及数据边界见[总索引](README.md)。',
            '', '## 编辑说明与谱系','',
            '本报告把12个campaign及一个未启动预检的科学说明汇编为一份可独立阅读的公开版本，',
            '依据本地162页总报告、协议、实际配置、机器结果及核验记录重新校对。它**不是**原PDF/Markdown的逐字公开副本。',
            '私人通信/截图、连接凭据及机器登录信息不公开；科学构想以独立叙述表达。',
            '原本地Markdown SHA-256：`'+origsha+'`。原件保持不变。',
            '本次编辑还区分了F原始点估计和bootstrap分布均值，并以实际数组/记录校正统计重复数。',
            '历史更正不改写原结果，集中见[CORRECTIONS](CORRECTIONS.md)。',
            '', '早期背景：L64冻结采样器修正后仍需跨尺寸验证；L128长程相关过强；',
            'Stage2B错误unit-coordinate对照与旧GO需要更正，不能把早期筛选当多seed确认。',
            '后续研究因此逐步从条件利用几何、上下文一致性、生成确认，转向训练size/spacing和精确局部必要能力。','']
    for c in C:
        base=OUT/c['slug']; t=(base/'README.md').read_text(encoding='utf-8')
        t=re.sub(r'^(#{1,5}) ',lambda m:'#'+m.group(1)+' ',t,flags=re.M)
        def fix(m):
            prefix,target=m.groups()
            if target.startswith(('http://','https://','#','mailto:')):return m.group()
            pathpart,sep,anchor=target.partition('#')
            return prefix+'('+relative((base/pathpart).resolve(),OUT)+(sep+anchor if sep else '')+')'
        t=re.sub(r'(!?\[[^\]]+\])\(([^)]+)\)',fix,t)
        report.append(t+'\n\n---\n')
    write(OUT/'REPORT_ZH.md','\n'.join(report))
    write(OUT/'report-editorial-provenance.json',json.dumps(dict(original_report=original.name,original_sha256=origsha,
        public_edition='REPORT_ZH.md',basis='campaign README assembly after machine-evidence review',
        excluded_categories=['private correspondence and screenshots','credentials and infrastructure access','unrelated work'],
        numerical_policy='No scientific arrays altered; F original estimate versus bootstrap mean and actual repetition counts explicitly distinguished'),ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
