import anafora
import anafora.validate

def test_schema_validate():
    schema = anafora.validate.Schema(anafora.ElementTree.fromstring("""
        <schema>
        <defaultattribute>
            <required>True</required>
        </defaultattribute>
        <definition>
            <entities>
                <entity type="X">
                        <properties>
                                <property type="A" input="choice">x,y</property>
                                <property type="B" />
                                <property type="C" instanceOf="Y,Z" />
                        </properties>
                </entity>
                <entity type="Y" />
                <entity type="Z" />
            </entities>
            <relations>
                <relation type="R">
                    <properties>
                        <property type="D" instanceOf="X" required="False" />
                        <property type="E" instanceOf="Y,Z" required="False" />
                    </properties>
                </relation>
            </relations>
        </definition>
        </schema>
        """))
    data = anafora.AnaforaData()
    entity1 = anafora.AnaforaEntity()
    entity1.id = "@1@"
    entity1.type = "X"
    entity1.properties["A"] = "x"
    data.annotations.append(entity1)
    assert schema.errors(data)
    entity1.properties["B"] = "y"
    assert schema.errors(data)
    entity1.properties["C"] = "z"
    assert schema.errors(data)
    entity2 = anafora.AnaforaEntity()
    entity2.id = "@2@"
    entity2.type = "X"
    data.annotations.append(entity2)
    entity1.properties["C"] = entity2
    assert schema.errors(data)
    entity2.type = "Y"
    assert not schema.errors(data)
    entity1.properties["A"] = "y"
    assert not schema.errors(data)
    entity1.properties["A"] = "z"
    assert schema.errors(data)
    entity1.properties["A"] = "x"
    assert not schema.errors(data)

    relation = anafora.AnaforaRelation()
    relation.id = "@3@"
    relation.type = ""
    data.annotations.append(relation)
    assert schema.errors(data)
    relation.type = "R"
    assert not schema.errors(data)
    relation.properties["D"] = entity1
    assert not schema.errors(data)
    relation.properties["E"] = entity1
    assert schema.errors(data)
    relation.properties["E"] = entity2
    assert not schema.errors(data)
    relation.properties["X"] = "Y"
    assert schema.errors(data)
