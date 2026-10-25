"""Preserve existing worksheet XML when selecting or appending result sheets."""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import csv, hashlib, json, posixpath, shutil, zipfile
from lxml import etree as ET

MAIN='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG='http://schemas.openxmlformats.org/package/2006/relationships'
CT='http://schemas.openxmlformats.org/package/2006/content-types'
TABLE_NAMES=['T01_Comparator_Overview','T02_Dataset_Composition','T03_Experimental_Configuration',
 'T04_Ablation_and_Baselines','T05_Standard_Test','T06_Robustness_Shifts','T07_Calibration_Trust',
 'T08_Coverage_Efficiency','T09_Statistical_Comparisons','T10_Calibration_Diagnosis','T11_RAID_Validation','T12_Computational_Cost']
def q(name):return '{'+MAIN+'}'+name
def xml(element):return ET.tostring(element,encoding='utf-8',xml_declaration=True)
def target_path(target):return target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target)
def sheet_paths(z):
    rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    targets={r.attrib['Id']:target_path(r.attrib['Target']) for r in rels}
    book=ET.fromstring(z.read('xl/workbook.xml'))
    return {s.attrib['name']:targets[s.attrib['{'+REL+'}id']] for s in book.find(q('sheets'))}

def select_sheets(source,destination,names):
    """Select sheets by reference, copying their original cells/styles verbatim."""
    source,destination=Path(source),Path(destination)
    with zipfile.ZipFile(source) as zin:
        paths=sheet_paths(zin);assert set(names)<=set(paths)
        book=ET.fromstring(zin.read('xl/workbook.xml'))
        sheets=book.find(q('sheets'))
        keep_ids=set()
        for s in list(sheets):
            if s.attrib['name'] not in names:sheets.remove(s)
            else:keep_ids.add(s.attrib['{'+REL+'}id'])
        for view in book.findall(q('bookViews')+'/'+q('workbookView')):view.set('activeTab','0')
        rels=ET.fromstring(zin.read('xl/_rels/workbook.xml.rels'))
        removed=set(paths.values())-{paths[n] for n in names}
        for r in list(rels):
            if r.attrib.get('Type','').endswith('/worksheet') and r.attrib['Id'] not in keep_ids:rels.remove(r)
        types=ET.fromstring(zin.read('[Content_Types].xml'))
        for item in list(types):
            if item.attrib.get('PartName','').lstrip('/') in removed:types.remove(item)
        replacements={'xl/workbook.xml':xml(book),'xl/_rels/workbook.xml.rels':xml(rels),'[Content_Types].xml':xml(types)}
        with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as zout:
            for info in zin.infolist():
                if info.filename in removed:continue
                zout.writestr(info,replacements.get(info.filename,zin.read(info)))

def append_sheets_verbatim(source,donor,destination):
    """Append authored sheets while leaving every source worksheet byte unchanged."""
    with zipfile.ZipFile(source) as zin,zipfile.ZipFile(donor) as extra:
        existing=sheet_paths(zin);added=sheet_paths(extra)
        assert not set(existing)&set(added)
        a=ET.fromstring(zin.read('xl/styles.xml'));b=ET.fromstring(extra.read('xl/styles.xml'))
        def collection(root,name):
            child=root.find(q(name))
            if child is None:child=ET.SubElement(root,q(name),{'count':'0'})
            return child
        maps={}
        num_a=collection(a,'numFmts');num_b=collection(b,'numFmts')
        next_num=max([163]+[int(x.attrib['numFmtId']) for x in num_a])+1
        num_map={}
        for fmt in num_b:
            copy=deepcopy(fmt);num_map[int(fmt.attrib['numFmtId'])]=next_num;copy.set('numFmtId',str(next_num));next_num+=1;num_a.append(copy)
        num_a.set('count',str(len(num_a)))
        for name in ['fonts','fills','borders']:
            dest=collection(a,name);src=collection(b,name);offset=len(dest)
            maps[name]={i:offset+i for i in range(len(src))}
            dest.extend(deepcopy(list(src)));dest.set('count',str(len(dest)))
        for name in ['cellStyleXfs','cellXfs']:
            dest=collection(a,name);src=collection(b,name);offset=len(dest)
            maps[name]={i:offset+i for i in range(len(src))}
            for item in src:
                copy=deepcopy(item)
                for attr,kind in [('fontId','fonts'),('fillId','fills'),('borderId','borders')]:
                    if attr in copy.attrib:copy.set(attr,str(maps[kind][int(copy.attrib[attr])]))
                n=int(copy.attrib.get('numFmtId','0'))
                if n in num_map:copy.set('numFmtId',str(num_map[n]))
                if name=='cellXfs' and 'xfId'in copy.attrib:copy.set('xfId',str(maps['cellStyleXfs'][int(copy.attrib['xfId'])]))
                dest.append(copy)
            dest.set('count',str(len(dest)))
        strings=[]
        if 'xl/sharedStrings.xml'in extra.namelist():strings=list(ET.fromstring(extra.read('xl/sharedStrings.xml')))
        book=ET.fromstring(zin.read('xl/workbook.xml'));sheets=book.find(q('sheets'))
        rels=ET.fromstring(zin.read('xl/_rels/workbook.xml.rels'));types=ET.fromstring(zin.read('[Content_Types].xml'))
        sheet_id=max(int(s.attrib['sheetId']) for s in sheets)
        used_rel={r.attrib['Id'] for r in rels};used_names=set(zin.namelist());new_files={}
        rel_index=1
        for name,path in added.items():
            sheet_id+=1
            while f'rId{rel_index}'in used_rel:rel_index+=1
            rid=f'rId{rel_index}';used_rel.add(rid)
            index=sheet_id
            new_path=f'xl/worksheets/sheet{index}.xml'
            while new_path in used_names:index+=1;new_path=f'xl/worksheets/sheet{index}.xml'
            used_names.add(new_path)
            sheet=ET.fromstring(extra.read(path))
            for c in sheet.iter(q('c')):
                c.set('s',str(maps['cellXfs'][int(c.attrib.get('s','0'))]))
                if c.attrib.get('t')=='s':
                    value=c.find(q('v'));si=strings[int(value.text)];c.remove(value);c.set('t','inlineStr')
                    inline=ET.SubElement(c,q('is'));inline.extend(deepcopy(list(si)))
            for elem in list(sheet.iter(q('col')))+list(sheet.iter(q('row'))):
                attr='style' if elem.tag==q('col') else 's'
                if attr in elem.attrib:elem.set(attr,str(maps['cellXfs'][int(elem.attrib[attr])]))
            assert sheet.find(q('drawing')) is None and sheet.find(q('tableParts')) is None
            ET.SubElement(sheets,q('sheet'),{'name':name,'sheetId':str(sheet_id),'{'+REL+'}id':rid})
            ET.SubElement(rels,'{'+PKG+'}Relationship',{'Id':rid,'Type':REL+'/worksheet','Target':new_path[3:]})
            ET.SubElement(types,'{'+CT+'}Override',{'PartName':'/'+new_path,'ContentType':'application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml'})
            new_files[new_path]=xml(sheet)
        replacements={'xl/workbook.xml':xml(book),'xl/styles.xml':xml(a),'xl/_rels/workbook.xml.rels':xml(rels),'[Content_Types].xml':xml(types)}
        with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as zout:
            for info in zin.infolist():zout.writestr(info,replacements.get(info.filename,zin.read(info)))
            for path,data in new_files.items():zout.writestr(path,data)
    return list(added)

def export_preserved_report(workbook,output_dir,skip_figures=False,source_figures=None,version='2.0.2'):
    """Keep supplied table values authoritative; do not rerun their calculations."""
    import openpyxl
    workbook,output_dir=Path(workbook).resolve(),Path(output_dir).resolve();output_dir.mkdir(parents=True,exist_ok=True)
    destination=output_dir/'RIGAT_Actual_Report.xlsx'
    if workbook==destination:raise ValueError('Choose a report directory different from the input workbook')
    select_sheets(workbook,destination,['README']+TABLE_NAMES)
    wb=openpyxl.load_workbook(workbook,read_only=True,data_only=False)
    table_counts={}
    for name in TABLE_NAMES:
        with (output_dir/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);count=0
            for row in wb[name].values:writer.writerow(row);count+=1
            table_counts[name]=max(0,count-1)
    wb.close()
    candidates=[Path(source_figures)] if source_figures else [workbook.parent,workbook.parent.parent/'RIGAT_Actual_Report']
    figures=[]
    if not skip_figures:
        for directory in candidates:
            for source in sorted(directory.glob('Fig*.png')):
                target=output_dir/source.name
                if source.resolve()!=target.resolve():shutil.copy2(source,target)
                if source.name not in figures:figures.append(source.name)
    def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest={'version':version,'source_workbook':str(workbook),'sha256':digest(workbook),
        'operation':'preserve_supplied_tables','tables':TABLE_NAMES,'table_rows':table_counts,
        'figures':figures,'recomputed_existing_values':False,'output_sha256':digest(destination)}
    (output_dir/'report_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'Preserved-table report complete: {output_dir}; {len(TABLE_NAMES)} tables',flush=True)
    return output_dir
