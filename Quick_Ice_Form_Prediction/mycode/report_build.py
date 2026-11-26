"""按 cn-academic-docx 规范生成实验报告 DOCX。"""
import json
from pathlib import Path

from PIL import Image
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parent
N = json.loads((ROOT / "report_numbers.json").read_text(encoding="utf-8"))
A = N["assets"]
OUT = ROOT.parent / "结冰冰形快速预测实验报告.docx"

doc = Document()

sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.5)
sec.top_margin = sec.bottom_margin = Cm(2.5)

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")


def run_font(run, size=12, bold=False, east="宋体"):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.bold = bold
    rPr = run._element.get_or_add_rPr()
    rPr.rFonts.set(qn("w:eastAsia"), east)



def stack(paths, out):
    """把多张图纵向拼成一张，用于附录。"""
    ims = [Image.open(str(x)).convert("RGB") for x in paths]
    w = max(im.width for im in ims)
    h = sum(im.height for im in ims) + 16 * (len(ims) - 1)
    canvas = Image.new("RGB", (w, h), "white")
    y = 0
    for im in ims:
        canvas.paste(im, ((w - im.width) // 2, y))
        y += im.height + 16
    canvas.save(str(out))
    return out

def body(text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    ind = p._p.get_or_add_pPr().get_or_add_ind()
    ind.set(qn("w:firstLineChars"), "200")
    ind.set(qn("w:firstLine"), "480")
    run_font(p.add_run(text), 12)
    return p


def h1(text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(12)
    pf.space_after = Pt(6)
    pf.keep_with_next = True
    run_font(p.add_run(text), 14, True)
    return p


def h2(text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(6)
    pf.space_after = Pt(3)
    pf.keep_with_next = True
    run_font(p.add_run(text), 12, True)
    return p


def caption(text, before=3, after=3, keep_next=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.keep_with_next = keep_next
    run_font(p.add_run(text), 10.5)
    return p


def note(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6)
    run_font(p.add_run(text), 9)
    return p


def fig(path, width=15.2):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width))


def _line(el, tag, sz):
    b = OxmlElement(tag)
    b.set(qn("w:val"), "single")
    b.set(qn("w:sz"), str(sz))
    b.set(qn("w:space"), "0")
    b.set(qn("w:color"), "000000")
    el.append(b)


def table(headers, rows, widths, cap, fs=10.5, note_text=None):
    caption(cap, before=8, after=3, keep_next=True)
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False

    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    _line(borders, "w:top", 12)
    _line(borders, "w:bottom", 12)
    for side in ("w:left", "w:right", "w:insideV", "w:insideH"):
        b = OxmlElement(side)
        b.set(qn("w:val"), "none")
        b.set(qn("w:sz"), "0")
        borders.append(b)
    tblPr.insert_element_before(borders, "w:shd", "w:tblLayout", "w:tblCellMar",
                                "w:tblLook", "w:tblCaption", "w:tblDescription")

    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.insert_element_before(layout, "w:tblCellMar", "w:tblLook",
                                "w:tblCaption", "w:tblDescription")

    grid = t._tbl.tblGrid
    for i, w in enumerate(widths):
        grid.gridCol_lst[i].set(qn("w:w"), str(int(w * 567)))
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Cm(w)

    hdr = t.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        run_font(p.add_run(h), fs, True)
        tcPr = cell._tc.get_or_add_tcPr()
        tcb = OxmlElement("w:tcBorders")
        _line(tcb, "w:bottom", 6)
        tcPr.insert_element_before(tcb, "w:shd", "w:noWrap", "w:tcMar",
                                   "w:textDirection", "w:tcFitText", "w:vAlign",
                                   "w:hideMark")

    for r in rows:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            run_font(p.add_run(str(v)), fs)

    for row in t.rows:
        rp = row._tr.get_or_add_trPr()
        rp.append(OxmlElement("w:cantSplit"))
    for row in t.rows[:-1]:
        for cell in row.cells:
            for par in cell.paragraphs:
                par.paragraph_format.keep_with_next = True
    trPr = hdr._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trPr.append(th)

    if note_text:
        note(note_text)


def equation(xml):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    p._p.append(parse_xml(xml))


def sub(base, sub_):
    return ('<m:sSub><m:e><m:r><m:t>%s</m:t></m:r></m:e>'
            '<m:sub><m:r><m:t>%s</m:t></m:r></m:sub></m:sSub>' % (base, sub_))


def nary(lo, hi, body_xml):
    return ('<m:nary><m:naryPr><m:chr m:val="\u2211"/><m:limLoc m:val="undOvr"/>'
            '<m:subHide m:val="0"/><m:supHide m:val="0"/></m:naryPr>'
            '<m:sub><m:r><m:t>%s</m:t></m:r></m:sub>'
            '<m:sup><m:r><m:t>%s</m:t></m:r></m:sup>'
            '<m:e>%s</m:e></m:nary>' % (lo, hi, body_xml))


EQ1 = ('<m:oMath ' + nsdecls("m", "w") + '>'
       '<m:r><m:t>x = \u03bc + </m:t></m:r>'
       + nary("i=1", "k", sub("a", "i") + sub("v", "i"))
       + '</m:oMath>')

EQ2 = ('<m:oMath ' + nsdecls("m", "w") + '>'
       '<m:r><m:t>f(T) = </m:t></m:r>'
       + nary("i=1", "K", sub("\u03b1", "i")
              + '<m:r><m:t>K(</m:t></m:r>' + sub("T", "i")
              + '<m:r><m:t>, T) + b</m:t></m:r>')
       + '</m:oMath>')

# ---------------- 标题区 ----------------
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(3)
run_font(p.add_run("基于POD降维与代理模型的翼型结冰冰形快速预测"), 15, True)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(2)
run_font(p.add_run("——SVM参数自对比与SVM/MLP对比"), 12)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(10)
run_font(p.add_run("《人工智能与航空应用》课程实验报告　　2026年11月"), 10.5)

h2("摘要")
body("以5个结冰温度下的翼型冰形为训练样本，采用本征正交分解将640维几何压缩为3个模态，"
     "累计能量95.26%，再用支持向量回归与多层感知机分别建立温度到模态系数的映射，"
     "在258、260、264 K三个温度上检验。SVM参数自对比表明：ε在10⁻⁴~10⁻²区间对精度无影响，"
     "C由0.1增至100使RMSE降低38.2%；SVM与MLP对比表明：MLP的MAE与RMSE分别低32.1%与10.2%，"
     "最大误差高11.0%。两种模型的几何误差均在1.0×10⁻³量级，与3模态截断误差同阶。")
body("关键词：结冰冰形；POD降维；支持向量回归；多层感知机；参数敏感性")

h1("一、引言")
body("结冰冰形由CFD数值模拟给出，单次计算需要求解流场、水滴轨迹与结冰热力学，耗时以小时计。"
     "在防除冰系统设计与容冰适航评估中，需要反复改变温度、液态水含量、结冰时间等条件，"
     "逐点调用CFD不可行。降阶模型把高维几何压到低维模态空间，再用代理模型拟合工况到模态系数的映射，"
     "可将单次预测压缩到毫秒级。")
body("本文以5个温度下的冰形为样本，完成POD降维、SVR与MLP建模，并回答两个问题："
     "SVR的C与ε如何影响精度；在同样的降维与测试条件下，SVR与MLP谁更合适。"
     "样本量只有5个，这一条件决定了模型容量的上限，也是全文分析的基准。")

h1("二、方法")
h2("2.1 数据与预处理")
body("训练集为257、259、261、263、265 K五个冰形，测试集为258、260、264 K三个冰形，"
     "每个冰形由320个点组成，坐标以弦长为尺度（弦长0.533）。冰形相对干净翼型的最大增量"
     "为0.002~0.004，约占0.3%~0.7%弦长，且集中在前缘±0.01范围，其余轮廓与翼型重合（表1、图1）。")
table(["用途", "温度 / K", "样本数", "点数", "冰厚 / 弦长"],
      [["训练", "257 / 259 / 261 / 263 / 265", "5", "320", "0.002~0.004"],
       ["测试", "258 / 260 / 264", "3", "320", "0.002~0.004"]],
      [2.6, 6.6, 2.2, 2.0, 3.4],
      "表1 数据清单",
      note_text="注：冰厚指冰形相对干净翼型的最大法向增量，按弦长归一。")
fig(A["data"], 13.0)
caption("图1 冰形与干净翼型的前缘对比")

body("读入时以列数判据跳过数据头，非数值分量按上下行均值补齐；几何异常按“相邻点间距超过中位间距3倍”"
     "判定并只替换发生跳变的分量。259 K样本中第175行的x越界与第188行的非数值分量由此定位并修复，"
     "修复后五个样本的点数、范围与连续性检验全部通过。")

h2("2.2 POD降维")
body("将每个冰形拉平为640维向量，减平均形状后做奇异值分解，按能量排序取前k个模态。"
     "任一冰形可表示为")
equation(EQ1)
body("式中μ为平均形状，vᵢ为第i个模态，aᵢ为对应系数。取k=3时累计能量95.26%，"
     "五个训练样本的重构最大误差为1.145×10⁻³。模态能量分布见表2与图2。")
table(["模态", "特征值", "能量占比", "累计占比"],
      [[str(i + 1), "%.4e" % N["lam"][i], "%.2f%%" % (N["ratio"][i] * 100),
        "%.2f%%" % (N["cum"][i] * 100)] for i in range(5)],
      [2.6, 5.0, 4.2, 4.2],
      "表2 模态能量分布")
fig(A["energy"], 10.5)
caption("图2 模态能量占比与累计曲线")

h2("2.3 回归模型")
body("SVR对每个模态系数单独建模，共3个回归器，核函数取RBF，其预测形式为")
equation(EQ2)
body("式中K(Tᵢ,T)=exp(−γ‖Tᵢ−T‖²)，αᵢ为对偶系数且0≤αᵢ≤C，b为截距。"
     "C控制超带惩罚，ε控制不敏感带宽度，γ控制核的影响半径。输入温度与输出系数均做标准化，"
     "γ固定为0.1。")
body("MLP采用1-16-16-3结构，激活函数tanh，求解器lbfgs，L2正则系数1×10⁻³，随机种子42。"
     "与SVR不同，MLP一次输出3个系数，隐层在三个模态之间共享。整体流程见图3。")
fig(A["flow"], 10.5)
caption("图3 技术路线")

h2("2.4 评价指标")
body("几何误差按四种口径统计：MAE、RMSE、最大误差与相对L2。前三个以弦长为尺度，"
     "相对L2以整条坐标向量的范数为分母。另设“平均形状”基线：不区分温度、直接输出训练集平均冰形，"
     "用于判断模型是否提供了有效信息。")

h1("三、SVM参数自对比")
h2("3.1 实验设置")
body("核函数固定为RBF，γ=0.1，C取0.1、1、100，ε取1×10⁻⁴、1×10⁻²，两两组合共6组；"
     "每组在5个训练温度上拟合、在3个测试温度上预测，统计三个测试样本的平均误差。")

h2("3.2 结果")
body("六组参数的测试误差见表3，RMSE与最大误差的对比见图4。")
c = N["contrast"]
table(["C", "ε", "MAE", "RMSE", "最大误差"],
      [["%g" % r[0], "%g" % r[1], "%.3e" % r[2], "%.3e" % r[3], "%.3e" % r[4]] for r in c],
      [3.0, 3.4, 3.6, 3.6, 3.6],
      "表3 SVM参数组合的测试误差",
      note_text="注：表中为258、260、264 K三个测试样本的平均值，单位为弦长。")
fig(A["contrast"], 12.5)
caption("图4 不同C与ε下的RMSE与最大误差")

h2("3.3 分析")
body("ε在10⁻⁴~10⁻²区间几乎不改变结果。同一C下两组误差的差值不超过4×10⁻⁶："
     "例如C=1时MAE为4.082×10⁻⁵与4.074×10⁻⁵，RMSE为1.595×10⁻⁴与1.592×10⁻⁴。"
     "原因是系数标准化后相邻训练点的距离在0.7~1.4之间，而ε最大只有1×10⁻²，"
     "误差带远窄于点间距，没有样本落入带内，对偶系数全部非零，ε因而失效。")
body("C是主导参数。C从0.1增至100，RMSE由2.100×10⁻⁴降到1.298×10⁻⁴，降幅38.2%；"
     "MAE由5.860×10⁻⁵降到2.671×10⁻⁵，降幅54.4%。C=0.1时对偶系数被压在上限0.1以内，"
     "模型欠拟合；C≥1后权重放开，精度随C提高。C由1增至100，RMSE再降18.6%，收益递减。")
body("最大误差不随C改善，始终在1.75×10⁻³~1.97×10⁻³之间，C=1时最小（1.751×10⁻³），"
     "C=100时反而回升到1.785×10⁻³。最大误差出现在前缘冰角，该处由第4模态描述，"
     "而k=3的截断误差已达1.145×10⁻³，回归器无法弥补被丢弃的几何分量。")
body("按RMSE选型，最优组合为C=100、ε=1×10⁻²；按最大误差选型，C=1最优。"
     "两者相差0.35×10⁻⁴（RMSE）与0.03×10⁻³（最大误差），量级相同，说明在5个样本下"
     "参数的精细调节空间有限。后续对比取C=10、γ=0.1、ε=1×10⁻³，处于两组最优之间。")

h1("四、SVM与MLP对比")
h2("4.1 实验设置")
body("两种模型共用同一组POD基（k=3）、同一批测试样本与同一套误差口径。"
     "SVM取C=10、γ=0.1、ε=1×10⁻³；MLP取上述结构，随机种子42，单次训练。"
     "两者输出均为3个模态系数，经逆变换得到320点冰形。")

h2("4.2 结果")
body("逐温度误差见表4，两种模型的RMSE与最大误差对比见图5。")
svm, nn = N["svm"], N["nn"]
rows = []
for i, t in enumerate(N["test_T"]):
    rows.append(["%d K" % t, "SVM", "%.2e" % svm[i][0], "%.2e" % svm[i][1],
                 "%.2e" % svm[i][2], "%.2f%%" % (svm[i][3] * 100)])
    rows.append(["", "MLP", "%.2e" % nn[i][0], "%.2e" % nn[i][1],
                 "%.2e" % nn[i][2], "%.2f%%" % (nn[i][3] * 100)])
rows.append(["平均", "SVM", "%.2e" % N["svm_avg"][0], "%.2e" % N["svm_avg"][1],
             "%.2e" % N["svm_avg"][2], "%.2f%%" % (N["svm_avg"][3] * 100)])
rows.append(["", "MLP", "%.2e" % N["nn_avg"][0], "%.2e" % N["nn_avg"][1],
             "%.2e" % N["nn_avg"][2], "%.2f%%" % (N["nn_avg"][3] * 100)])
table(["温度", "模型", "MAE", "RMSE", "最大误差", "相对L2"], rows,
      [2.2, 2.0, 3.4, 3.4, 3.6, 2.8],
      "表4 SVM与MLP的测试误差",
      note_text="注：误差单位为弦长；相对L2以整条640维坐标向量为分母。")
fig(A["cmp"], 12.5)
caption("图5 SVM与MLP的RMSE与最大误差对比")

h2("4.3 分析")
body("MLP在平均误差上占优：MAE为1.90×10⁻⁵对2.80×10⁻⁵，低32.1%；"
     "RMSE为1.15×10⁻⁴对1.28×10⁻⁴，低10.2%。三个测试温度上MLP的RMSE均低于SVM，"
     "方向一致。MLP一次输出3个系数，隐层共享，三个模态之间的相关性被同一组权重利用；"
     "SVR对每个模态独立拟合，模态间的协同信息无法进入模型。三个测试温度的曲线对比见图6~图8。")
body("SVM在最大误差上占优：1.57×10⁻³对1.74×10⁻³，低11.0%。最大误差集中在冰角尖端，"
     "该点由系数组合决定，对单个系数的偏差高度敏感。三个独立SVR的误差不相关，极值被平均掉；"
     "MLP的三个输出共享隐层，一个输出在极值处偏大时其余输出同步偏移，最大误差随之放大。")
body("两种模型相对平均形状基线均有明显改进。基线RMSE为2.28×10⁻⁴、最大误差为2.12×10⁻³；"
     "SVM将RMSE降低43.9%、最大误差降低25.9%，MLP分别降低49.6%与17.8%。"
     "RMSE的改进幅度大于最大误差，说明模型对整体冰形把握较好，对冰角局部仍有欠缺。")
body("两种模型的误差同为1.0×10⁻³量级，与k=3的POD截断误差1.145×10⁻³同阶。"
     "回归器的差异只影响10%~30%的分项指标，量级由模态数与样本量决定。")
fig(A["three_258"], 11.5)
caption("图6 258 K：真实冰形与两种模型预测对比")
fig(A["three_260"], 11.5)
caption("图7 260 K：真实冰形与两种模型预测对比")
fig(A["three_264"], 11.5)
caption("图8 264 K：真实冰形与两种模型预测对比")

h1("五、讨论")
body("误差可分解为两部分：POD截断与回归。取k=3时截断误差为1.145×10⁻³，"
     "模型总最大误差为1.57×10⁻³（SVM）与1.74×10⁻³（MLP），回归环节贡献约0.4×10⁻³~0.6×10⁻³。"
     "若取k=4可精确重构几何，但系数维度增至4，5个样本下每个SVR只有4个自由度，过拟合风险同步上升。")
body("留一验证给出同量级的结果：SVM的RMSE为1.95×10⁻⁴、最大误差1.783×10⁻³，"
     "MLP为1.74×10⁻⁴与1.632×10⁻³。留一误差略高于测试误差，因为留出端点温度时预测需要外推半步，"
     "而三个测试温度均位于训练区间内部。两组结果一致，说明模型没有依赖特定测试点。")
body("单样本预测（系数回归加逆变换）耗时约0.8 ms，与CFD单次冰形计算的小时级相比低6个数量级以上。"
     "这一差距是降阶模型在工程上可用的前提。")
body("适用条件由数据决定：翼型与来流条件固定，温度限定在257~265 K，输出为内插结果。"
     "温度超出该区间时RBF核值衰减到零，预测退化为常数，不能作为外推依据。"
     "扩展到其他翼型或工况，需要按翼型、按条件补充样本并重新训练POD基。")

h1("六、结论")
body("1. 5个训练样本的POD基取3个模态可保留95.26%的能量，五个冰形的重构最大误差为1.145×10⁻³。")
body("2. SVR的ε在10⁻⁴~10⁻²区间不起作用，误差差异小于4×10⁻⁶；C主导精度，由0.1增至100使RMSE降低38.2%，"
     "但最大误差停留在1.75×10⁻³~1.97×10⁻³，其下限由POD截断决定。")
body("3. 在相同POD基与测试样本下，MLP的MAE、RMSE分别比SVM低32.1%、10.2%，最大误差高11.0%；"
     "两者相对平均形状基线将RMSE降低43.9%与49.6%。整体误差由3个模态与5个样本决定，"
     "模型选择只带来10%~30%的分项改进。")
body("4. 单样本预测耗时约0.8 ms，可用于257~265 K区间内的冰形快速估算；提高精度需增加样本或保留第4模态。")

h1("参考文献")
refs = [
    "《基于降维与支持向量机的快速冰形预测》实验指导书. 2026.",
    "申晓斌, 郁嘉, 林贵平, 等. 基于特征正交分解法的翼型结冰冰形快速预测[J]. 航空动力学报, 2013, 28(4): 807-812.",
    "刘藤, 李栋, 黄冉冉, 等. 基于降阶模型的翼型结冰冰形预测方法[J]. 北京航空航天大学学报, 2019, 45(5): 1033-1041.",
    "余婧, 蒋安林, 刘亮, 等. 基于PCA降维的气动外形参数化方法[J]. 航空学报, 2024, 45(10): 129125.",
    "Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine learning in Python[J]. "
    "Journal of Machine Learning Research, 2011, 12: 2825-2830.",
]
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.left_indent = Cm(0.74)
    pf.first_line_indent = Cm(-0.74)
    run_font(p.add_run("[%d] %s" % (i, r)), 10.5)

# ---------------- 附录 ----------------
FIG_SRC = ROOT / "figs_contrast"
app_svm = stack([ROOT / "figs_SVM" / ("compare_SVM_%d.png" % t) for t in (258, 260, 264)],
                ROOT / "figs_report" / "appendix_svm.png")
app_nn = stack([ROOT / "figs_nn" / ("compare_nn_%d.png" % t) for t in (258, 260, 264)],
               ROOT / "figs_report" / "appendix_nn.png")

h = h1("附录A SVM预测结果（258 / 260 / 264 K）")
h.paragraph_format.page_break_before = True
fig(app_svm, 14.8)
caption("图A1 SVM在三个测试温度下的预测与逐点误差")

h = h1("附录B MLP预测结果（258 / 260 / 264 K）")
h.paragraph_format.page_break_before = True
fig(app_nn, 14.8)
caption("图B1 MLP在三个测试温度下的预测与逐点误差")

footer = sec.footer
fp = footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fp.add_run("— ")
run_font(r, 9)
r = fp.add_run()
fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = "PAGE"
fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "end")
r._r.append(fld1); r._r.append(instr); r._r.append(fld2)
run_font(r, 9)
r = fp.add_run(" —")
run_font(r, 9)

doc.save(OUT)
print("saved:", OUT)
