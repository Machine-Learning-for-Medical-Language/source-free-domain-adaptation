import anafora
import anafora.regex


def test_regex_annotator():
    annotator = anafora.regex.RegexAnnotator({
        'aa+': ('A', {}),
        'a': ('A', {'X': '2'}),
        'bb': ('B', {'Y': '1'})
    })
    text = "bb aaa"
    data = anafora.AnaforaData()
    annotator.annotate(text, data)

    assert len(list(data.annotations)) == 2
    [b_annotation, a_annotation] = data.annotations
    assert b_annotation.type == "B"
    assert b_annotation.spans == ((0, 2),)
    assert dict(b_annotation.properties.items()) == {'Y': '1'}
    assert a_annotation.type == "A"
    assert a_annotation.spans == ((3, 6),)
    assert dict(a_annotation.properties.items()) == {}


def test_preannotated():
    annotator = anafora.regex.RegexAnnotator({
        'aa+': ('A', {'X': '2'}),
        'a': ('A', {}),
        'bb': ('B', {'Y': '1'})
    }, {
        'C': {'Z': '3'}
    })
    text = "bb aaa"
    data = anafora.AnaforaData()
    bb = anafora.AnaforaEntity()
    bb.id = "1@preannotated"
    bb.type = "B"
    bb.spans = ((0, 2),)
    data.annotations.append(bb)
    aaa = anafora.AnaforaEntity()
    aaa.id = "2@preannotated"
    aaa.type = "C"
    aaa.spans = ((3, 6),)
    data.annotations.append(aaa)
    annotator.annotate(text, data)

    assert len(list(data.annotations)) == 3
    [b_annotation, c_annotation, a_annotation] = data.annotations
    assert b_annotation.type == "B"
    assert b_annotation.spans == ((0, 2),)
    assert dict(b_annotation.properties.items()) == {'Y': '1'}
    assert c_annotation.type == "C"
    assert c_annotation.spans == ((3, 6),)
    assert dict(c_annotation.properties.items()) == {'Z': '3'}
    assert a_annotation.type == "A"
    assert a_annotation.spans == ((3, 6),)
    assert dict(a_annotation.properties.items()) == {'X': '2'}


def test_many_groups():
    regex_predictions = {}
    for i in range(1, 1000):
        regex_predictions['a' * i] = ('A' * i, {})
    annotator = anafora.regex.RegexAnnotator(regex_predictions)
    text = "aaaaaaaaaa"
    data = anafora.AnaforaData()
    annotator.annotate(text, data)

    assert len(list(data.annotations)) == 1
    [annotation] = data.annotations
    assert annotation.type == "AAAAAAAAAA"
    assert annotation.spans == ((0, 10),)
    assert dict(annotation.properties.items()) == {}


def test_file_roundtrip(tmpdir):
    annotator_path = str(tmpdir.join("temp.annotator"))
    annotator = anafora.regex.RegexAnnotator({
        'the year': ('DATE', {}),
        'John': ('PERSON', {'type': 'NAME', 'gender': 'MALE'}),
        '.1.2.\d+;': ('OTHER', {})
    }, {
        'PERSON': {'type': 'NAME', 'gender': 'FEMALE'}
    })
    annotator.to_file(annotator_path)
    assert anafora.regex.RegexAnnotator.from_file(annotator_path) == annotator


def test_simple_file(tmpdir):
    path = tmpdir.join("temp.annotator")
    path.write("""\
aaa aaa\tA
b\tB\t{"x": "y"}
\\dc\\s+x\tC
""")
    annotator = anafora.regex.RegexAnnotator({
        'aaa aaa': ('A', {}),
        'b': ('B', {'x': 'y'}),
        r'\dc\s+x': ('C', {})
    })
    assert anafora.regex.RegexAnnotator.from_file(str(path)) == annotator


def test_train():
    text1 = "aaa bb ccccc dddd"
    data1 = anafora.AnaforaData(anafora.ElementTree.fromstring("""
    <data>
        <annotations>
            <entity>
                <id>1</id>
                <type>AA</type>
                <span>0,6</span><!-- "aaa bb" -->
                <properties>
                    <a>A</a>
                </properties>
            </entity>
            <entity>
                <id>2</id>
                <type>AA</type>
                <span>7,12</span><!-- "ccccc" -->
                <properties>
                    <c>B</c>
                </properties>
            </entity>
            <entity>
                <id>3</id>
                <type>EMPTY</type>
                <span>0,0</span>
            </entity>
        </annotations>
    </data>
    """))
    text2 = "ccccc dddd ccccc dddd ccccc."
    data2 = anafora.AnaforaData(anafora.ElementTree.fromstring("""
    <data>
        <annotations>
            <entity>
                <id>1</id>
                <type>CC</type>
                <span>0,5</span><!-- "ccccc" -->
                <properties>
                    <c>B</c>
                </properties>
            </entity>
            <entity>
                <id>2</id>
                <type>CC</type>
                <span>11,16</span><!-- "ccccc" -->
                <properties>
                    <c>C</c>
                </properties>
            </entity>
            <entity>
                <id>3</id>
                <type>CC</type>
                <span>22,27</span><!-- "ccccc" -->
                <properties>
                    <c>C</c>
                    <d>D</d>
                </properties>
            </entity>
            <entity>
                <id>4</id>
                <type>PERIOD</type>
                <span>27,28</span><!-- "." -->
            </entity>
        </annotations>
    </data>
    """))

    annotator = anafora.regex.RegexAnnotator({
        r'\baaa\s+bb\b': ('AA', {"a": "A"}),
        r'\bccccc\b': ('CC', {"c": "C", "d": "D"}),
        r'\b\.': ('PERIOD', {}),
    }, {
        'AA': {'a': 'A', 'c': 'B'},
        'CC': {'c': 'C', 'd': 'D'},
    })
    assert anafora.regex.RegexAnnotator.train([(text1, data1), (text2, data2)]) == annotator


def test_filter_by_precision():
    annotator = anafora.regex.RegexAnnotator({
        r'the': ("THE", {}),
        r'\bthe\b': ("THE", {}),
        r'yer\b': ("ER", {}),
        r'er\b': ("ER", {})
    })
    text = "the theater near the record player"
    data = anafora.AnaforaData(anafora.ElementTree.fromstring("""
    <data>
        <annotations>
            <entity>
                <id>1</id>
                <type>THE</type>
                <span>0,3</span><!-- "the" -->
            </entity>
            <entity>
                <id>2</id>
                <type>THE</type>
                <span>17,20</span><!-- "the" -->
            </entity>
            <entity>
                <id>3</id>
                <type>ER</type>
                <span>9,11</span><!-- "er" -->
            </entity>
            <entity>
                <id>4</id>
                <type>ER</type>
                <span>32,34</span><!-- "." -->
            </entity>
        </annotations>
    </data>
    """))
    annotator.prune_by_precision(0.6, [(text, data)])
    assert annotator == anafora.regex.RegexAnnotator({
        r'the': ("THE", {}),
        r'\bthe\b': ("THE", {}),
        r'er\b': ("ER", {})
    })
    annotator.prune_by_precision(1.0, [(text, data)])
    assert annotator == anafora.regex.RegexAnnotator({
        r'\bthe\b': ("THE", {}),
        r'er\b': ("ER", {})
    })
