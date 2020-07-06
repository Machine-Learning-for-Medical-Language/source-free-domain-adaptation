import anafora
import anafora.heuristic


def test_relation_to_closest():
    def get_xml():
        return anafora.ElementTree.fromstring("""
            <data>
                <annotations>
                    <entity>
                        <id>1</id>
                        <span>0,5</span>
                        <type>X</type>
                    </entity>
                    <entity>
                        <id>2</id>
                        <span>15,20</span>
                        <type>X</type>
                    </entity>
                    <entity>
                        <id>3</id>
                        <span>25,30</span>
                        <type>X</type>
                    </entity>
                    <entity>
                        <id>4</id>
                        <span>0,3</span>
                        <type>Y</type>
                    </entity>
                    <entity>
                        <id>5</id>
                        <span>21,24</span>
                        <type>Y</type>
                    </entity>
                    <entity>
                        <id>6</id>
                        <span>35,40</span>
                        <type>Y</type>
                    </entity>
                </annotations>
            </data>
            """)

    data = anafora.AnaforaData(get_xml())
    z1 = anafora.AnaforaRelation()
    z1.id = "7"
    data.annotations.append(z1)
    z1.type = "Z"
    z1.properties["source"] = data.annotations.select_id("1")
    z1.properties["target"] = data.annotations.select_id("4")
    z1.properties["foo"] = "bar"

    z2 = anafora.AnaforaRelation()
    z2.id = "8"
    data.annotations.append(z2)
    z2.type = "Z"
    z2.properties["source"] = data.annotations.select_id("2")
    z2.properties["target"] = data.annotations.select_id("5")
    z2.properties["foo"] = "bar"

    z3 = anafora.AnaforaRelation()
    z3.id = "9"
    data.annotations.append(z3)
    z3.type = "Z"
    z3.properties["source"] = data.annotations.select_id("3")
    z3.properties["target"] = data.annotations.select_id("5")
    z3.properties["foo"] = "bar"

    data = anafora.AnaforaData(get_xml())
    anafora.heuristic.add_relations_to_closest(data, "X", "Y", "Z", "source", "target", [("foo", "bar")])
    assert set(data.annotations.select_type("Z")) == {z1, z2, z3}

    # make sure it doesn't fail with 0 source and 0 target annotations
    data = anafora.AnaforaData(get_xml())
    anafora.heuristic.add_relations_to_closest(data, "A", "Y", "Z", "source", "target")
    data = anafora.AnaforaData(get_xml())
    anafora.heuristic.add_relations_to_closest(data, "X", "B", "Z", "source", "target")
