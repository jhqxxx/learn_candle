use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};

// 定义段落结构
#[derive(Debug, Deserialize)]
#[allow(unused)]
struct Paragraph {
    #[serde(rename = "行号")]
    line_number: i64,

    #[serde(rename = "是否重复")]
    is_duplicate: bool,

    #[serde(rename = "是否跨文件重复")]
    is_cross_file_duplicate: bool,

    #[serde(rename = "md5")]
    md5_hash: String,

    #[serde(rename = "内容")]
    content: String,
}

// 定义文件记录结构
#[derive(Debug, Deserialize)]
#[allow(unused)]
struct FileRecord {
    #[serde(rename = "文件名")]
    filename: String,

    #[serde(rename = "是否待查文件")]
    needs_check: bool,

    #[serde(rename = "是否重复文件")]
    is_duplicate_file: bool,

    #[serde(rename = "文件大小")]
    file_size: i64,

    #[serde(rename = "simhash")]
    simhash: u64,

    #[serde(rename = "最长段落长度")]
    max_paragraph_length: i64,

    #[serde(rename = "段落数")]
    paragraph_count: i64,

    #[serde(rename = "去重段落数")]
    deduplicated_paragraph_count: i64,

    #[serde(rename = "低质量段落数")]
    low_quality_paragraph_count: i64,

    #[serde(rename = "段落")]
    paragraphs: Vec<Paragraph>,
}

#[allow(unused)]
pub fn get_mnbvc_string() -> String {
    let file = File::open("/mnt/d/huggingface_dataset/wiki/0.jsonl").expect("file open error");
    let reader = BufReader::new(file);
    let mut all_line_str = String::new();
    let mut i = 0;
    for line in reader.lines() {
        i += 1;
        if i < 100 {
            continue;
        }
        let line_str = line.expect("reader line error");
        let record: FileRecord = serde_json::from_str(&line_str).expect("str to json error");
        let mut line_content = String::new();
        for para in record.paragraphs.iter() {
            line_content += &para.content;
            line_content += "\n";
        }
        all_line_str += &line_content;
        all_line_str += "<|endoftext|>";
        if i > 10000 {
            break;
        }
    }
    println!("total line number: {}", i);
    all_line_str
}

#[allow(unused)]
pub fn write_file(file_path: &str, string: String) {
    let file = File::create(file_path).expect("create file error");
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{}", &string).expect("write string error");
    println!("成功写入 {:?}", file_path);
}

#[allow(unused)]
pub fn read_txt(txt_path: &str) -> String {
    let mut file = File::open(txt_path).expect("open file error");
    let mut content = String::new();
    let _ = file.read_to_string(&mut content);
    content
}
