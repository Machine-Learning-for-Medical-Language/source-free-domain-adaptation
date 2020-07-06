import pytest

import anafora


def test_empty():
    data = anafora.AnaforaData(anafora.ElementTree.fromstring('<data/>'))
    assert list(data.annotations) == []

    data = anafora.AnaforaData(anafora.ElementTree.fromstring('<data><annotations></annotations></data>'))
    assert list(data.annotations) == []


def test_duplicate_id():
    with pytest.raises(ValueError):
        anafora.AnaforaData(anafora.ElementTree.fromstring('''
        <data>
            <annotations>
                <entity><id>1</id></entity>
                <entity><id>1</id></entity>
            </annotations>
        </data>'''))

    data = anafora.AnaforaData()
    entity1 = anafora.AnaforaEntity()
    entity1.id = "1"
    entity2 = anafora.AnaforaEntity()
    entity2.id = "1"
    data.annotations.append(entity1)
    with pytest.raises(ValueError):
        data.annotations.append(entity2)


def test_add_entity():
    data = anafora.AnaforaData()
    assert str(data) == '<data />'
    entity = anafora.AnaforaEntity()
    with pytest.raises(ValueError) as exception_info:
        data.annotations.append(entity)
    assert "id" in str(exception_info.value)
    assert str(data) == '<data />'
    entity.id = "1"
    data.annotations.append(entity)
    assert str(data) == '<data><annotations><entity><id>1</id></entity></annotations></data>'
    entity.type = "X"
    entity.parents_type = "Y"
    entity.properties["name1"] = "value1"
    assert str(data) == ('<data><annotations><entity>' +
                         '<id>1</id>' +
                         '<type>X</type>' +
                         '<parentsType>Y</parentsType>' +
                         '<properties><name1>value1</name1></properties>' +
                         '</entity></annotations></data>')
    del entity.properties["name1"]
    assert str(data) == ('<data><annotations><entity>' +
                         '<id>1</id>' +
                         '<type>X</type>' +
                         '<parentsType>Y</parentsType>' +
                         '</entity></annotations></data>')
    with pytest.raises(ValueError):
        del entity.properties["name2"]


def test_add_reference():
    data = anafora.AnaforaData()
    entity1 = anafora.AnaforaEntity()
    entity1.id = "@1@"
    entity2 = anafora.AnaforaEntity()
    entity2.id = "@2@"
    with pytest.raises(ValueError) as exception_info:
        entity2.properties["link"] = entity1
    assert "<annotations" in str(exception_info.value)
    data.annotations.append(entity1)
    with pytest.raises(ValueError):
        entity2.properties["link"] = entity1
    assert "<annotations" in str(exception_info.value)
    data.annotations.append(entity2)
    entity2.properties["link"] = entity1
    assert str(data) == ('<data><annotations>' +
                         '<entity><id>@1@</id></entity>' +
                         '<entity><id>@2@</id><properties><link>@1@</link></properties></entity>' +
                         '</annotations></data>')


def test_remove():
    data = anafora.AnaforaData()
    assert str(data) == '<data />'
    entity1 = anafora.AnaforaEntity()
    entity1.id = "@1@"
    data.annotations.append(entity1)
    entity2 = anafora.AnaforaEntity()
    entity2.id = "@2@"
    entity2.properties["name"] = "value"
    data.annotations.append(entity2)
    assert list(data.annotations) == [entity1, entity2]
    assert str(data) == ('<data><annotations>' +
                         '<entity><id>@1@</id></entity>' +
                         '<entity><id>@2@</id><properties><name>value</name></properties></entity>' +
                         '</annotations></data>')
    data.annotations.remove(entity1)
    assert list(data.annotations) == [entity2]
    assert str(data) == ('<data><annotations>' +
                         '<entity><id>@2@</id><properties><name>value</name></properties></entity>' +
                         '</annotations></data>')
    data.annotations.remove(entity2)
    assert list(data.annotations) == []
    assert str(data) == '<data><annotations /></data>'


def test_spans():
    data = anafora.AnaforaData(anafora.ElementTree.fromstring('''
        <data>
            <annotations>
                <relation>
                    <id>1</id>
                    <type>R1</type>
                    <properties>
                        <relation>2</relation>
                    </properties>
                </relation>
                <relation>
                    <id>2</id>
                    <type>R2</type>
                    <properties>
                        <entity>3</entity>
                    </properties>
                </relation>
                <entity>
                    <id>3</id>
                    <type>E1</type>
                    <span>5,7</span>
                </entity>
            </annotations>
        </data>'''))
    assert data.annotations.select_id("1").spans == ((((5, 7),),),)
    assert data.annotations.select_id("2").spans == (((5, 7),),)
    assert data.annotations.select_id("3").spans == ((5, 7),)


def test_recursive_entity():
    data = anafora.AnaforaData()
    entity = anafora.AnaforaEntity()
    entity.id = "@1@"
    data.annotations.append(entity)
    entity.properties["self"] = entity
    assert entity.is_self_referential()
    assert data.annotations.find_self_referential().id == entity.id

    data = anafora.AnaforaData()
    a = anafora.AnaforaEntity()
    a.id = "A"
    data.annotations.append(a)
    b = anafora.AnaforaEntity()
    b.id = "B"
    data.annotations.append(b)
    c = anafora.AnaforaEntity()
    c.id = "C"
    data.annotations.append(c)
    d = anafora.AnaforaEntity()
    d.id = "D"
    data.annotations.append(d)
    b.properties["x"] = a
    c.properties["y"] = a
    d.properties["1"] = b
    d.properties["2"] = c
    assert not d.is_self_referential()



def test_sort():
    data = anafora.AnaforaData(anafora.ElementTree.fromstring('''
        <data>
            <annotations>
                <entity>
                    <id>1</id>
                    <type>E</type>
                    <span>5,7</span>
                </entity>
                <entity>
                    <id>2</id>
                    <type>E</type>
                    <span>3,4</span>
                </entity>
            </annotations>
        </data>'''))
    assert [a.id for a in sorted(data.annotations)] == ['2', '1']

