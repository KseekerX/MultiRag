import os
from pathlib import Path
import json
from collections import defaultdict
import asyncio
from image_utils.async_image_analysis import AsyncImageAnalysis
from dotenv import load_dotenv
load_dotenv()

def parse_all_pdfs(datas_dir, output_base_dir):
    """
    步骤1：解析所有PDF，输出内容到 data_base_json_content/
    """
    from mineru_parse_pdf import do_parse
    datas_dir = Path(datas_dir)
    output_base_dir = Path(output_base_dir)
    pdf_files = list(datas_dir.rglob('*.pdf'))
    if not pdf_files:
        print(f"未找到PDF文件于: {datas_dir}")
        return
    for pdf_path in pdf_files:
        file_name = pdf_path.stem
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        # 为每个PDF文件创建对应的输出子目录
        output_dir = output_base_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        # 调用do_parse函数解析PDF文件，输出内容列表JSON文件
        do_parse(
            output_dir=str(output_dir),
            pdf_file_names=[file_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["ch"],
            backend="pipeline",
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=False,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=True
        )
        print(f"已输出: {output_dir / 'auto' / (file_name + '_content_list.json')}")

# 把具有相同page_idx的项目归类到一起，比如同一页的文本、图片
def group_by_page(content_list):
    # 创建一个默认值为列表的字典pages
    pages = defaultdict(list)
    for item in content_list:
        # 获取每个项目的页面索引（默认为0）
        page_idx = item.get('page_idx', 0)
        # 将项目添加到对应页面索引的列表中
        pages[page_idx].append(item)
    # 返回转换为普通字典的结果
    return dict(pages)


def item_to_markdown(item, enable_image_caption=True):
    """
    将不同类型的文档内容项（文本、图像或表格）转换为 Markdown 格式
    enable_image_caption: 是否启用多模态视觉分析（图片caption补全），默认True。
    """
    # 默认API参数：硅基流动Qwen/Qwen2.5-VL-32B-Instruct
    vision_provider = "modelscope"
    vision_model = "Qwen/Qwen2.5-VL-7B-Instruct"
    vision_api_key = os.getenv("MS_API_KEY")
    vision_base_url = os.getenv("MS_BASE_URL")
    

    if item['type'] == 'text':  # 文本类型处理
        # 根据 text_level 确定标题级别
        # level == 1 生成一级标题（# text）
        # level == 2 生成二级标题（## text）
        # 其他情况生成普通段落文本
        level = item.get('text_level', 0)
        text = item.get('text', '')
        if level == 1:
            return f"# {text}\n\n"
        elif level == 2:
            return f"## {text}\n\n"
        else:
            return f"{text}\n\n"
    elif item['type'] == 'image':  # 图像类型处理
        # 尝试从 image_caption 字段获取现有的图像描述
        captions = item.get('image_caption', [])
        caption = captions[0] if captions else ''
        img_path = item.get('img_path', '')
        # 如果没有caption，且允许视觉分析，调用多模态API补全
        if enable_image_caption and not caption and img_path and os.path.exists(img_path):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                async def get_caption():
                    # 使用异步方式调用 AsyncImageAnalysis 分析图像
                    async with AsyncImageAnalysis(
                        provider=vision_provider,
                        api_key=vision_api_key,
                        base_url=vision_base_url,
                        vision_model=vision_model
                    ) as analyzer:
                        # 从分析结果中提取标题或描述作为图像的 caption
                        result = await analyzer.analyze_image(local_image_path=img_path)
                        return result.get('title') or result.get('description') or ''
                caption = loop.run_until_complete(get_caption())
                loop.close()
                if caption:
                    # 更新 item 中的 image_caption 字段
                    item['image_caption'] = [caption]
            except Exception as e:
                print(f"图片解释失败: {img_path}, {e}")
        # 生成 Markdown 格式的图像标签：![caption](img_path)
        md = f"![{caption}]({img_path})\n"
        return md + "\n"
    elif item['type'] == 'table':  # 表格类型处理
        captions = item.get('table_caption', [])  # 获取表格的标题（caption）
        caption = captions[0] if captions else ''
        table_html = item.get('table_body', '')  # 获取表格的 HTML 内容（table_body）
        img_path = item.get('img_path', '')  
        md = ''
        if caption:
            md += f"**{caption}**\n"  # 如果有标题，则以粗体形式添加
        if img_path:
            md += f"![{caption}]({img_path})\n"  # 如果有图像路径，则添加图像
        md += f"{table_html}\n\n"  # 添加表格的 HTML 内容
        return md
    else: # 对于不支持的类型，返回一个空行。
        return '\n'

# 
def assemble_pages_to_markdown(pages):
    """
    用于将按页面分组的内容转换为 Markdown 格式。
    pages: 字典，键是页面索引（page index），值是该页面上所有内容项（items）的列表。
    """
    page_md = {} # 创建一个空字典 page_md，用于存储所有页的 Markdown 内容。
    # 按页面排序顺序遍历所有页面索引。
    for page_idx in sorted(pages.keys()):
        md = ''  # 初始化一个空字符串 md 来累积该页面的 Markdown 内容
        for item in pages[page_idx]:  # 遍历该页面上的所有内容项（items）
            # 对每个内容项调用 item_to_markdown 函数将其转换为 Markdown 格式
            md += item_to_markdown(item, enable_image_caption=True)
        # 将转换结果追加到 md 字符串中
        page_md[page_idx] = md
    return page_md
    
def process_all_pdfs_to_page_json(input_base_dir, output_base_dir):
    """
    步骤2：将 content_list.json 转为按页面组织的 page_content.json
    input_base_dir：输入目录路径，包含第一步生成的 content_list.json 文件
    output_base_dir：输出目录路径，用于存放转换后的 page_content.json 文件
    """
    input_base_dir = Path(input_base_dir)
    output_base_dir = Path(output_base_dir)
    # 获取输入目录下所有子目录的列表，这些子目录对应于每个PDF文件的处理结果
    pdf_dirs = [d for d in input_base_dir.iterdir() if d.is_dir()]
    # 遍历每个PDF文件对应的目录
    for pdf_dir in pdf_dirs:
        file_name = pdf_dir.name
        # 首先在 pdf_dir/auto/ 目录下查找 content_list.json 文件
        json_path = pdf_dir / 'auto' / f'{file_name}_content_list.json'
        # 如果没找到，则在 pdf_dir/file_name/auto/ 目录下查找
        if not json_path.exists():
            sub_dir = pdf_dir / file_name
            json_path2 = sub_dir / 'auto' / f'{file_name}_content_list.json'
            if json_path2.exists():
                json_path = json_path2
            else:
                # 如果两个位置都找不到文件，则输出错误信息并跳过该PDF
                print(f"未找到: {json_path} 也未找到: {json_path2}")
                continue
        # 打开并加载 content_list.json 文件，该文件包含PDF中提取的所有内容项（文本、图像、表格等）。
        with open(json_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)
        # 调用 group_by_page 函数，将内容项按页面索引进行分组，使同一页的内容项归类到一起。
        pages = group_by_page(content_list)
        # 调用 assemble_pages_to_markdown 函数，将按页面分组的内容转换为Markdown格式。
        page_md = assemble_pages_to_markdown(pages)
        output_dir = output_base_dir / file_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json_path = output_dir / f'{file_name}_page_content.json'
        # 将转换后的页面Markdown内容保存为JSON文件
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(page_md, f, ensure_ascii=False, indent=2)
        print(f"已输出: {output_json_path}")

def process_page_content_to_chunks(input_base_dir, output_json_path):
    """
    步骤3：将第二步生成的所有 page_content.json 合并为统一的 JSON 文件 all_pdf_page_chunks.json
    input_base_dir：输入目录路径，包含第二步生成的所有 page_content.json 文件
    output_json_path：输出文件路径，用于保存合并后的所有页面内容
    """
    input_base_dir = Path(input_base_dir)
    # 初始化一个空列表 all_chunks，用于存储所有PDF的所有页面内容
    all_chunks = []
    # 遍历输入目录中的所有项目，只处理目录（即每个PDF文件对应的目录），跳过非目录项目。
    for pdf_dir in input_base_dir.iterdir():
        if not pdf_dir.is_dir():
            continue
        file_name = pdf_dir.name
        # 首先在 pdf_dir/ 目录下查找 page_content.json 文件
        page_content_path = pdf_dir / f"{file_name}_page_content.json"
        # 如果没找到，则在 pdf_dir/file_name/ 目录下查找（处理可能存在的嵌套目录结构）
        if not page_content_path.exists():
            sub_dir = pdf_dir / file_name
            page_content_path2 = sub_dir / f"{file_name}_page_content.json"
            if page_content_path2.exists():
                page_content_path = page_content_path2
            else:
                # 如果两个位置都找不到文件，则输出错误信息并跳过该PDF
                print(f"未找到: {page_content_path} 也未找到: {page_content_path2}")
                continue
        # 打开并加载 page_content.json 文件，该文件包含特定PDF的所有页面内容
        with open(page_content_path, 'r', encoding='utf-8') as f:
            page_dict = json.load(f)
        """
        对于每个页面，创建一个"chunk"对象，包含：
            id：唯一标识符，格式为 pdf文件名_page_页面索引
            content：页面的Markdown内容
            metadata：元数据，包括页面索引和原始PDF文件名
        """
        for page_idx, content in page_dict.items():
            chunk = {
                "id": f"{file_name}_page_{page_idx}",
                "content": content,
                "metadata": {
                    "page": page_idx,
                    "file_name": file_name + ".pdf"
                }
            }
            all_chunks.append(chunk)
    # 将所有chunks保存到指定的输出文件中，使用UTF-8编码和2个空格缩进，确保非ASCII字符正确保存。
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"已输出: {output_json_path}")

def main():
    base_dir = Path(__file__).parent  # 脚本所在目录，作为所有相对路径的基准
    datas_dir = base_dir / 'datas'  # 输入目录，存放待处理的PDF文件，路径为 ./datas
    content_dir = base_dir / 'data_base_json_content'  # 第一步输出目录，存放解析后的 content_list.json 文件
    page_dir = base_dir / 'data_base_json_page_content'  # 第二步输出目录，存放按页面组织的 page_content.json 文件
    chunk_json_path = base_dir / 'all_pdf_page_chunks.json'  # 最终输出文件路径，存放合并后的所有页面内容
    # 步骤1：PDF → content_list.json
    parse_all_pdfs(datas_dir, content_dir)
    # 步骤2：content_list.json → page_content.json
    process_all_pdfs_to_page_json(content_dir, page_dir)
    # 步骤3：page_content.json → all_pdf_page_chunks.json
    process_page_content_to_chunks(page_dir, chunk_json_path)
    print("全部处理完成！")

if __name__ == '__main__':
    main()
