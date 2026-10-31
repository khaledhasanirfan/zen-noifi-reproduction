"""Generate measured charts, a concise PDF, and slide talking points with Python."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader, PdfWriter
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from common import ROOT, save_csv

rl_config.invariant = 1
OUT = ROOT/'output'
FIG = OUT/'figures'
PDF = OUT/'pdf/experiment_report.pdf'
NAVY, TEAL, ORANGE = '#18364B', '#168B89', '#DA7B3F'


def read_json(path):
    value = json.loads(path.read_text())
    if value.get('status') != 'completed':
        raise ValueError(f'Experiment not completed: {path}')
    return value


def read_csv(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'Empty experiment result: {path}')
    return rows


def save_figure(name):
    plt.savefig(FIG/f'{name}.png', dpi=220, bbox_inches='tight', facecolor='white')
    plt.savefig(FIG/f'{name}.svg', bbox_inches='tight', metadata={'Date': None})
    svg = FIG/f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close()


def main():
    nroot = ROOT/'results/noifi-basic'; zroot = ROOT/'results/zen-attribution'
    noifi = read_json(nroot/'summary.json')
    zen = read_json(zroot/'summary.json')
    patch = read_json(ROOT/'results/zen-patch-demo/summary.json')
    rows = read_csv(nroot/'detection_metrics.csv')
    attacks = read_csv(nroot/'attack_metrics.csv')
    predictions = read_csv(zroot/'predictions.csv')
    coverage = read_csv(zroot/'artifact_coverage.csv')
    stability = read_csv(nroot/'stability.csv')
    FIG.mkdir(parents=True, exist_ok=True); PDF.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': NAVY, 'text.color': NAVY,
                         'axes.titleweight': 'bold', 'figure.dpi': 140})
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['attack'], row['detector'], int(row['n']))].append(row)
    summary = []
    for (attack, detector, n), group in grouped.items():
        summary.append({'attack': attack, 'detector': detector, 'n': n,
                        'auroc_mean': mean(float(r['auroc']) for r in group),
                        'auroc_std': pstdev(float(r['auroc']) for r in group),
                        'recall_mean': mean(float(r['recall_at_calibrated_threshold']) for r in group),
                        'test_fpr_mean': mean(float(r['observed_test_fpr']) for r in group),
                        'ms_per_sample_mean': mean(float(r['ms_per_sample']) for r in group)})
    timing_low = mean(r['ms_per_sample_mean'] for r in summary if r['detector']=='NoiFi MED' and r['n']==1)
    timing_high = mean(r['ms_per_sample_mean'] for r in summary if r['detector']=='NoiFi MED' and r['n']==20)
    save_csv(nroot/'comparison.csv', summary)
    # Paired attack/detector comparison. Accuracy and AUROC are kept separate.
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    names = [r['attack'] for r in attacks]
    positions = np.arange(len(names))
    for offset, detector, n, color in [(-0.18, 'Reconstruction MSE', 0, ORANGE),
                                      (0.18, 'NoiFi MED', 5, TEAL)]:
        values = [next(r['auroc_mean'] for r in summary if r['attack'] == a and r['detector'] == detector and r['n'] == n) for a in names]
        bars = ax.bar(positions+offset, values, 0.34, label=detector+(' (N=5)' if n else ''), color=color)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set(xticks=positions, xticklabels=names, ylim=(0, 1.13), ylabel='Detection AUROC', title='Reduced real-data experiment: detector comparison')
    ax.axhline(0.5, ls=':', color='#AAB7C0', lw=1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    fig.tight_layout(); save_figure('noifi_comparison')
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for attack_name, color in zip(names, [TEAL, ORANGE]):
        group = sorted([r for r in summary if r['attack'] == attack_name and r['detector'] == 'NoiFi MED'], key=lambda r: r['n'])
        ns = [r['n'] for r in group]
        axes[0].errorbar(ns, [r['auroc_mean'] for r in group], yerr=[r['auroc_std'] for r in group], marker='o', capsize=3, color=color, label=attack_name)
        axes[1].plot(ns, [r['ms_per_sample_mean'] for r in group], 'o-', color=color, label=attack_name)
        stab = [mean(float(r['mean_sample_score_std']) for r in stability if r['attack'] == attack_name and int(r['n']) == n) for n in ns]
        axes[2].plot(ns, stab, 'o-', color=color, label=attack_name)
    for ax, title, label in zip(axes, ['Detection', 'CPU batch timing', 'Score variability'],
                              ['AUROC (mean +/- SD)', 'Milliseconds per sample', 'Mean per-sample score SD']):
        ax.set(title=title, xlabel='Randomization count N', ylabel=label)
        ax.set_xticks([1, 5, 10, 20]); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); save_figure('noifi_n_study')
    # CMR attribution heat map, using authors' recovered artifacts.
    sims = read_csv(zroot/'similarity.csv')
    bases = zen['base_candidates']; queries = [r['query'] for r in predictions]
    matrix = np.array([[next(float(r['cmr_similarity']) for r in sims if r['query'] == q and r['base'] == b) for b in bases] for q in queries])
    fig, ax = plt.subplots(figsize=(8, 4.9))
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set(xticks=range(len(bases)), xticklabels=bases, yticks=range(len(queries)), yticklabels=queries,
           title='ZEN upstream attribution on supplied recovered artifacts')
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right', fontsize=9)
    for i in range(len(queries)):
        for j in range(len(bases)):
            ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if matrix[i,j] > 0.55 else NAVY)
    fig.colorbar(im, ax=ax, fraction=0.03, label='CMR similarity')
    fig.tight_layout(); save_figure('zen_attribution')
    fig, ax = plt.subplots(figsize=(7.5, 3))
    ranked = sorted(predictions, key=lambda r: float(r['margin']))
    ax.barh([r['query'] for r in ranked], [float(r['margin']) for r in ranked], color=TEAL)
    ax.set(xlabel='Top-one minus top-two CMR similarity', title='Simple ZEN extension: attribution ambiguity')
    fig.tight_layout(); save_figure('zen_margin')
    # Learning curves make the limitations of short training inspectable.
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    for ax, name in zip(axes, ['classifier', 'dae']):
        hist = read_csv(nroot/f'{name}_training.csv')
        ax.plot([int(r['epoch']) for r in hist], [float(r['train_loss']) for r in hist], label='Train', color=TEAL)
        ax.plot([int(r['epoch']) for r in hist], [float(r['validation_loss']) for r in hist], label='Validation', color=ORANGE)
        ax.set(title=name.upper(), xlabel='Epoch', ylabel='Loss'); ax.legend(frameon=False)
    fig.tight_layout(); save_figure('training_curves')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportTitle', fontName='Helvetica-Bold', fontSize=23, leading=28, textColor=colors.HexColor(NAVY), spaceAfter=12))
    styles.add(ParagraphStyle(name='SectionTitle', fontName='Helvetica-Bold', fontSize=16, leading=20, textColor=colors.HexColor(NAVY), spaceAfter=10))
    styles['BodyText'].fontSize=10; styles['BodyText'].leading=14; styles['BodyText'].spaceAfter=8
    styles.add(ParagraphStyle(name='SmallNote', fontName='Helvetica', fontSize=8, leading=11, textColor=colors.HexColor('#536776'), spaceAfter=7))
    story = []
    def p(text, style='BodyText'):
        story.append(Paragraph(text, styles[style]))
    def table(data, widths):
        formatted = [[Paragraph(str(v), styles['SmallNote']) for v in row] for row in data]
        t = Table(formatted, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E5F0F3')),
                              ('LINEBELOW',(0,0),(-1,0),0.8,colors.HexColor(TEAL)),
                              ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),
                              ('BOTTOMPADDING',(0,0),(-1,-1),6),
                              ('LINEBELOW',(0,1),(-1,-1),0.3,colors.HexColor('#DBE3E7'))]))
        story.extend([t, Spacer(1, 10)])
    def picture(name, width):
        from PIL import Image as PILImage
        path = FIG/f'{name}.png'
        with PILImage.open(path) as image:
            w, h = image.size
        story.extend([Image(str(path), width=width, height=width*h/w), Spacer(1, 8)])
    def newpage(title):
        story.append(PageBreak()); p(title, 'ReportTitle')
    p('ZEN + NoiFi', 'ReportTitle')
    p('Basic implementations, small extensions, measured comparisons', 'SectionTitle')
    p('Khaled Hasan Irfan | MacBook Air M5 | CPU | Python '+noifi['python'])
    table([['Experiment', 'What was executed', 'Outcome'],
           ['NoiFi basic', f'{noifi["recordings"]} real CSI recordings; compact CNN/DAE; FGSM + PGD', f'Clean test accuracy: {noifi["clean_accuracy"]:.1%}'],
           ['NoiFi extension', 'N = 1, 3, 5, 10, 20; three permutation seeds', 'AUROC, recall, test FPR, latency and score SD measured'],
           ['ZEN attribution', 'Upstream similarity code; supplied recovered artifacts', f'{zen["correct"]}/{zen["queries"]} base families identified'],
           ['ZEN patch demo', 'Local function bytecode; identical model weights; 64 inputs', f'Output error: {patch["after_max_absolute_error"]:.1f} after patch'],
           ['Full ZEN recovery', 'Supplied Linux memory dump + official YOLO patch inference', 'Not run: needs separate compatible setup and dump']],[90,255,160])
    p('Scope of the comparison', 'SectionTitle')
    p('ZEN attributes and restores model computation. NoiFi detects adversarial CSI inputs. These solve different problems, so their accuracy and AUROC are not ranked against each other. Each is compared with an appropriate baseline or ablation.')
    p('What changed from the initial project', 'SectionTitle')
    p('Installed Python 3.10; added a small runnable implementation, real-data download with hash checks, separate calibration/test partitions, fail-fast scripts, evidence CSVs, numerical output checking, and Python-generated plots/PDF. Removed completion claims based only on folders or logs.')
    p('All reported values come from the CSV/JSON files in results/. No paper headline scores are presented as local results.', 'SmallNote')
    newpage('NoiFi | basic experiment')
    split = noifi['config']['split_counts']
    p(f'Real RoboFiSense CSI, eight classes. One sample per recording. Partitions: {split["train"]} train, {split["validation"]} validation, {split["calibration"]} calibration, {split["test"]} final test. Inputs are normalized as in the upstream loader and resized to 64 x 64.')
    p('Mechanism: DAE reconstruction -> residual noise -> independent pixel permutations -> classifier penultimate features -> mean Euclidean distance (paper Eq. 8). The baseline uses reconstruction MSE without randomized features.')
    p(f'Clean test accuracy: {noifi["clean_accuracy"]:.1%}; uniform-noise control accuracy: {noifi["uniform_noise_control_accuracy"]:.1%}. The random control uses the same 0.10 amplitude bound; label preservation is not assumed.', 'SmallNote')
    picture('noifi_comparison', 500)
    data = [['Attack', 'ASR*', 'Detector', 'AUROC', 'Recall**', 'Test FPR']]
    for atk in attacks:
        for detector, n in [('Reconstruction MSE',0),('NoiFi MED',5)]:
            r = next(r for r in summary if r['attack']==atk['attack'] and r['detector']==detector and r['n']==n)
            data.append([atk['attack'],f'{float(atk["attack_success_rate_on_clean_correct"]):.1%}',detector+(' N=5' if n else ''),f'{r["auroc_mean"]:.3f}',f'{r["recall_mean"]:.1%}',f'{r["test_fpr_mean"]:.1%}'])
    table(data,[45,50,145,65,80,85])
    p('* ASR is measured only on clean-correct test recordings. Detection includes all test attacks. Both attacks use L-infinity epsilon = 0.10 on normalized inputs; PGD uses 20 steps and random start.', 'SmallNote')
    p('** Thresholds use a separate benign calibration partition with a 5% FPR target. The observed held-out test FPR is reported, not assumed to equal 5%. NoiFi rows average three permutation seeds; they share one trained model.', 'SmallNote')
    fg_mse = next(r for r in summary if r['attack']=='FGSM' and r['detector']=='Reconstruction MSE')
    fg_med = next(r for r in summary if r['attack']=='FGSM' and r['detector']=='NoiFi MED' and r['n']==5)
    pg_mse = next(r for r in summary if r['attack']=='PGD' and r['detector']=='Reconstruction MSE')
    pg_med = next(r for r in summary if r['attack']=='PGD' and r['detector']=='NoiFi MED' and r['n']==5)
    p(f'Observed comparison: FGSM AUROC is {fg_mse["auroc_mean"]:.3f} for MSE and {fg_med["auroc_mean"]:.3f} for MED; PGD is {pg_mse["auroc_mean"]:.3f} and {pg_med["auroc_mean"]:.3f}, respectively. This reduced run does not show a universal improvement from MED. Small models and resized inputs do not establish the full paper results.')
    newpage('NoiFi | small extension')
    p('Question: how many randomized references are needed before scores stabilize, and what is the cost? The trained CNN, DAE, recordings and adversarial examples remain fixed. Only N and permutation seed vary.')
    picture('noifi_n_study', 510)
    table([['N','Attack','AUROC mean +/- SD','Recall','Test FPR','ms/sample']]+[
        [r['n'],r['attack'],f'{r["auroc_mean"]:.3f} +/- {r["auroc_std"]:.2g}',f'{r["recall_mean"]:.1%}',f'{r["test_fpr_mean"]:.1%}',f'{r["ms_per_sample_mean"]:.2f}']
        for r in summary if r['detector']=='NoiFi MED'],[30,50,150,80,80,80])
    p('Timing includes DAE reconstruction, permutations, feature extraction and scoring. It is CPU wall time per sample amortized over batches of up to 32, not single-request latency. Initialization, model training and attack generation are excluded. These timings are hardware-specific.', 'SmallNote')
    p('Score SD is first computed across the three permutation seeds for each held-out sample, then averaged across samples and benign/adversarial kinds. It measures randomization stability, not uncertainty across independently trained models.', 'SmallNote')
    p(f'Observed trade-off: AUROC changes very little across N. Score variability decreases, while per-sample CPU time grows from about {timing_low:.2f} ms at N=1 to {timing_high:.2f} ms at N=20. These measurements support using fewer references for this reduced setup; broader validation is needed.', 'SmallNote')
    newpage('ZEN | attribution and patching')
    p(f'Executed the upstream ZEN similarity algorithm on 13 authors-supplied recovered CMR artifacts against seven base candidates. Top-one family identification: {zen["correct"]}/{zen["queries"]}. No new RAM image was recovered in this experiment.')
    p(f'Ablation comparison: mathematical representation alone (MR): {zen["mr_only_correct"]}/13; programmatic representation alone (PR): {zen["pr_only_correct"]}/13; combined representation (CMR): {zen["correct"]}/13. This is a closed-set attribution check on the supplied artifacts.', 'SmallNote')
    picture('zen_attribution', 470)
    missing = sum(int(r['missing_bytecode_files']) for r in coverage)
    p(f'Artifact coverage: {missing} referenced bytecode files are absent across the 20 inspected models. Upstream missing-file behavior is retained and counts are in artifact_coverage.csv. This limits the interpretation of the similarity scores.', 'SmallNote')
    p('Controlled patch demonstration', 'SectionTitle')
    p(f'With the same weights but the base forward function, maximum output error was {patch["before_max_absolute_error"]:.6f}. Restoring our locally captured custom forward bytecode reduced it to {patch["after_max_absolute_error"]:.1f} over 64 inputs. This illustrates code-plus-weights recovery; it is not the official YOLO memory experiment.')
    newpage('ZEN | extension and next steps')
    p('Simple extension: rank candidates and record the top-two similarity margin. A small margin flags cases worth inspecting; it is not a calibrated confidence score.')
    picture('zen_margin', 480)
    closest = min(predictions, key=lambda r: float(r['margin']))
    p(f'Smallest observed margin: {closest["query"]}, {float(closest["margin"]):.4f}; predicted family {closest["predicted_base"]}. All 13 supplied queries were still identified correctly in this closed candidate set.')
    p('What is still required', 'SectionTitle')
    p('Full ZEN reproduction requires the supplied LiME dump, the matched Linux x86-64 / Python 3.8.18 environment, Volatility recovery, and successful official YOLO inference after patching. None of those steps is marked complete here. The original-vs-recovered output checker is ready for real tensor exports.')
    p('Sources and download links', 'SectionTitle')
    for label, url in [('ZEN official code','https://github.com/CyFI-Lab-Public/ZEN'),
                       ('NoiFi official code','https://github.com/shahriar0651/NoiFi'),
                       ('RoboFiSense recordings','https://github.com/SiamiLab/RoboFiSense'),
                       ('ZEN supplied memory dump','https://drive.google.com/file/d/1nsTN716aGoT7AU32HmuFQjCYYXmakjHR/view')]:
        p(f'{label}: <link href="{url}" color="{TEAL}">{url}</link>','SmallNote')
    p('Paper methods checked against the provided Achieving Zen and Adversarial Noise PDFs. Source commits are recorded in reports/upstream-commits.txt and results/noifi-basic/config.json.', 'SmallNote')
    def footer(canvas, document):
        canvas.setStrokeColor(colors.HexColor('#DBE3E7')); canvas.line(44,40,551,40)
        canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#536776'))
        canvas.drawString(44,27,'ZEN + NoiFi | Reduced experiments and explicit limitations')
        canvas.drawRightString(551,27,str(document.page))
    SimpleDocTemplate(str(PDF), pagesize=(595.28,841.89), leftMargin=44, rightMargin=44,
                      topMargin=38, bottomMargin=52, title='ZEN and NoiFi - Basic Experiments',
                      author='Khaled Hasan Irfan').build(story, onFirstPage=footer, onLaterPages=footer)
    # Remove automatically generated creation/modification metadata too.
    reader=PdfReader(PDF); writer=PdfWriter()
    writer.append_pages_from_reader(reader)
    writer.add_metadata({'/Title':'ZEN and NoiFi - Basic Experiments','/Author':'Khaled Hasan Irfan'})
    with PDF.open('wb') as stream:
        writer.write(stream)
    notes = f'''# Slide talking points

1. **Problem:** ZEN studies model recovery/attribution; NoiFi studies adversarial CSI detection. Their metrics serve different tasks.
2. **Implemented:** Installed Python 3.10 and ran compact NoiFi NSM/NRM/NDM modules on {noifi['recordings']} real RoboFiSense recordings. Clean test accuracy: {noifi['clean_accuracy']:.1%}.
3. **Compared:** Reconstruction MSE vs NoiFi MED under FGSM and PGD, with a separate calibration split. Use figures/noifi_comparison.png and results/noifi-basic/comparison.csv.
4. **Extended NoiFi:** N=1,3,5,10,20 with three permutation seeds. Compare detection quality, CPU batch timing and score variability in figures/noifi_n_study.png.
5. **Executed ZEN stage:** Upstream attribution on supplied recovered CMRs identified {zen['correct']}/{zen['queries']} base families. Report incomplete upstream artifact coverage.
6. **Extended ZEN:** Used top-two attribution margin to expose ambiguous cases. Smallest margin: {closest['query']} ({float(closest['margin']):.4f}).
7. **Demonstrated patching:** On 64 controlled local inputs, output error changed from {patch['before_max_absolute_error']:.6f} to {patch['after_max_absolute_error']:.1f} after restoring custom forward bytecode. This is a mechanism demo.
8. **Limitations:** Compact 64x64 architecture, one model-training seed, digital white-box attacks and mixed-location recording split. Full ZEN RAM recovery and official YOLO patch inference were not run.

## Measured comparison to say aloud

- Reconstruction MSE versus NoiFi MED (N=5): FGSM AUROC {fg_mse['auroc_mean']:.3f} versus {fg_med['auroc_mean']:.3f}; PGD {pg_mse['auroc_mean']:.3f} versus {pg_med['auroc_mean']:.3f}. MED does not improve both attacks in this reduced run.
- N=1 to N=20 barely changes AUROC, reduces score variability, and raises CPU batch time from about {timing_low:.2f} to {timing_high:.2f} ms/sample.
- ZEN MR-only attribution identifies {zen['mr_only_correct']}/13 families, versus {zen['correct']}/13 for combined MR+PR. The missing artifact coverage limits the claim.

## Evidence

- Per-recording hashes and partitions: results/noifi-basic/recordings.csv
- Configuration and source commit: results/noifi-basic/config.json
- Raw scores and thresholds: results/noifi-basic/scores.csv and detection_metrics.csv
- ZEN candidates and coverage: results/zen-attribution/similarity.csv and artifact_coverage.csv
- All figures are generated by scripts/generate_report.py; editable SVGs accompany PNGs.
'''
    (OUT/'slide_notes.md').write_text(notes)
    print(f'Created {PDF} and {FIG}')


if __name__ == '__main__':
    main()
