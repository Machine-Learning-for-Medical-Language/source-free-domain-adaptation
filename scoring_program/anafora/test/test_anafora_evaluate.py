import pytest

import anafora
import anafora.evaluate


def test_score_data():
    reference = anafora.AnaforaData(anafora.ElementTree.fromstring("""
    <data>
        <annotations>
            <entity>
                <id>1</id>
                <span>0,5</span>
                <type>X</type>
            </entity>
            <entity>
                <id>2</id>
                <span>5,10</span>
                <type>Y</type>
            </entity>
            <entity>
                <id>3</id>
                <span>15,20</span>
                <type>Y</type>
            </entity>
            <relation>
                <id>4</id>
                <type>Z</type>
                <properties>
                    <Source>1</Source>
                    <Target>2</Target>
                    <Prop1>T</Prop1>
                    <Prop2>A</Prop2>
                </properties>
            </relation>
            <relation>
                <id>5</id>
                <type>Z</type>
                <properties>
                    <Source>2</Source>
                    <Target>3</Target>
                    <Prop1>T</Prop1>
                    <Prop2>B</Prop2>
                </properties>
            </relation>
        </annotations>
    </data>
    """))
    predicted = anafora.AnaforaData(anafora.ElementTree.fromstring("""
    <data>
        <annotations>
            <entity>
                <id>6</id><!-- different -->
                <span>0,5</span>
                <type>X</type>
            </entity>
            <entity>
                <id>7</id><!-- different -->
                <span>5,10</span>
                <type>X</type><!-- different -->
            </entity>
            <entity>
                <id>8</id><!-- different -->
                <span>15,20</span>
                <type>Y</type>
            </entity>
            <relation>
                <id>9</id><!-- different -->
                <type>Z</type>
                <properties>
                    <Source>6</Source>
                    <Target>7</Target>
                    <Prop1>T</Prop1>
                    <Prop2>A</Prop2>
                </properties>
            </relation>
            <relation>
                <id>10</id><!-- different -->
                <type>Z</type>
                <properties>
                    <Source>7</Source>
                    <Target>8</Target>
                    <Prop1>F</Prop1><!-- different -->
                    <Prop2>B</Prop2>
                </properties>
            </relation>
        </annotations>
    </data>
    """))
    named_scores = anafora.evaluate.score_data(reference, predicted)
    assert set(named_scores.keys()) == {
        "*", ("*", "<span>"),
        "X", ("X", "<span>"),
        "Y", ("Y", "<span>"),
        "Z", ("Z", "<span>"), ("Z", "Source"), ("Z", "Target"), ("Z", "Prop1"), ("Z", "Prop2"),
        ("Z", "Prop1", "T"), ("Z", "Prop1", "F"), ("Z", "Prop2", "A"), ("Z", "Prop2", "B"),
    }
    scores = named_scores["X"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 2
    scores = named_scores["X", "<span>"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 2
    scores = named_scores["Y"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 1
    scores = named_scores["Y", "<span>"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 1
    scores = named_scores["Z"]
    assert scores.correct == 0
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "<span>"]
    assert scores.correct == 2
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "Prop1"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "Prop1", "T"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 1
    scores = named_scores["Z", "Prop1", "F"]
    assert scores.correct == 0
    assert scores.reference == 0
    assert scores.predicted == 1
    scores = named_scores["Z", "Prop2"]
    assert scores.correct == 2
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "Prop2", "A"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["Z", "Prop2", "B"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["*"]
    assert scores.correct == 1 + 1 + 0
    assert scores.reference == 1 + 2 + 2
    assert scores.predicted == 2 + 1 + 2
    scores = named_scores["*", "<span>"]
    assert scores.correct == 1 + 1 + 2
    assert scores.reference == 1 + 2 + 2
    assert scores.predicted == 2 + 1 + 2

    named_scores = anafora.evaluate.score_data(reference, predicted, exclude=["X", "Y"])
    assert set(named_scores.keys()) == {
        "*", ("*", "<span>"),
        "Z", ("Z", "<span>"), ("Z", "Source"), ("Z", "Target"), ("Z", "Prop1"), ("Z", "Prop2"),
        ("Z", "Prop1", "T"), ("Z", "Prop1", "F"), ("Z", "Prop2", "A"), ("Z", "Prop2", "B"),
    }
    scores = named_scores["Z"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "<span>"]
    assert scores.correct == 2
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "Prop1"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "Prop1", "T"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 1
    scores = named_scores["Z", "Prop1", "F"]
    assert scores.correct == 0
    assert scores.reference == 0
    assert scores.predicted == 1
    scores = named_scores["Z", "Prop2"]
    assert scores.correct == 2
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "Prop2", "A"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["Z", "Prop2", "B"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["*"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["*", "<span>"]
    assert scores.correct == 2
    assert scores.reference == 2
    assert scores.predicted == 2

    named_scores = anafora.evaluate.score_data(reference, predicted, include=[("Z", "Prop1", "T")])
    assert set(named_scores.keys()) == {("Z", "Prop1", "T")}
    scores = named_scores["Z", "Prop1", "T"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 1

    named_scores = anafora.evaluate.score_data(reference, predicted, include=[("Z", "Prop1", "F")])
    assert set(named_scores.keys()) == {("Z", "Prop1", "F")}
    scores = named_scores["Z", "Prop1", "F"]
    assert scores.correct == 0
    assert scores.reference == 0
    assert scores.predicted == 1

    named_scores = anafora.evaluate.score_data(reference, predicted, include=["Z"], exclude=[("Z", "<span>")])
    assert set(named_scores.keys()) == {
        "Z", ("Z", "Source"), ("Z", "Target"), ("Z", "Prop1"), ("Z", "Prop2"),
        ("Z", "Prop1", "T"), ("Z", "Prop1", "F"), ("Z", "Prop2", "A"), ("Z", "Prop2", "B"),
    }
    scores = named_scores["Z"]
    assert scores.correct == 0
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "Prop1"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "Prop1", "T"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 1
    scores = named_scores["Z", "Prop1", "F"]
    assert scores.correct == 0
    assert scores.reference == 0
    assert scores.predicted == 1
    scores = named_scores["Z", "Prop2"]
    assert scores.correct == 2
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "Prop2", "A"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["Z", "Prop2", "B"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1


def test_score_data_overlap():
    # This test is identical to the one above except that the spans have been changed so that they're overlapping
    # instead of being exactly equal
    reference = anafora.AnaforaData(anafora.ElementTree.fromstring("""
    <data>
        <annotations>
            <entity>
                <id>1</id>
                <span>0,5</span>
                <type>X</type>
            </entity>
            <entity>
                <id>2</id>
                <span>5,10</span>
                <type>Y</type>
            </entity>
            <entity>
                <id>3</id>
                <span>15,20</span>
                <type>Y</type>
            </entity>
            <relation>
                <id>4</id>
                <type>Z</type>
                <properties>
                    <Source>1</Source>
                    <Target>2</Target>
                    <Prop1>T</Prop1>
                    <Prop2>A</Prop2>
                </properties>
            </relation>
            <relation>
                <id>5</id>
                <type>Z</type>
                <properties>
                    <Source>2</Source>
                    <Target>3</Target>
                    <Prop1>T</Prop1>
                    <Prop2>B</Prop2>
                </properties>
            </relation>
            <relation>
                <id>6</id>
                <type>Ref</type>
                <properties>
                    <Ref>1</Ref>
                </properties>
            </relation>
            <relation>
                <id>7</id>
                <type>Ref</type>
                <properties>
                    <Ref>6</Ref>
                </properties>
            </relation>
        </annotations>
    </data>
    """))
    predicted = anafora.AnaforaData(anafora.ElementTree.fromstring("""
    <data>
        <annotations>
            <entity>
                <id>6</id><!-- different -->
                <span>0,4</span>
                <type>X</type>
            </entity>
            <entity>
                <id>7</id><!-- different -->
                <span>6,10</span>
                <type>X</type><!-- different -->
            </entity>
            <entity>
                <id>8</id><!-- different -->
                <span>19,20</span>
                <type>Y</type>
            </entity>
            <relation>
                <id>9</id><!-- different -->
                <type>Z</type>
                <properties>
                    <Source>6</Source>
                    <Target>7</Target>
                    <Prop1>T</Prop1>
                    <Prop2>A</Prop2>
                </properties>
            </relation>
            <relation>
                <id>10</id><!-- different -->
                <type>Z</type>
                <properties>
                    <Target>8</Target>
                    <Source>7</Source>
                    <Prop2>B</Prop2>
                    <Prop1>F</Prop1><!-- different -->
                </properties>
            </relation>
        </annotations>
    </data>
    """))
    named_scores = anafora.evaluate.score_data(
        reference, predicted, spans_type=anafora.evaluate._OverlappingSpans)
    assert set(named_scores.keys()) == {
        "*", ("*", "<span>"),
        "X", ("X", "<span>"),
        "Y", ("Y", "<span>"),
        "Z", ("Z", "<span>"), ("Z", "Source"), ("Z", "Target"), ("Z", "Prop1"), ("Z", "Prop2"),
        ("Z", "Prop1", "T"), ("Z", "Prop1", "F"), ("Z", "Prop2", "A"), ("Z", "Prop2", "B"),
        "Ref", ("Ref", "<span>"), ("Ref", "Ref"),
        }
    scores = named_scores["X"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 2
    scores = named_scores["X", "<span>"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 2
    scores = named_scores["Y"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 1
    scores = named_scores["Y", "<span>"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 1
    scores = named_scores["Z"]
    assert scores.correct == 0
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "<span>"]
    assert scores.correct == 2
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "Prop1"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "Prop1", "T"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 1
    scores = named_scores["Z", "Prop1", "F"]
    assert scores.correct == 0
    assert scores.reference == 0
    assert scores.predicted == 1
    scores = named_scores["Z", "Prop2"]
    assert scores.correct == 2
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "Prop2", "A"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["Z", "Prop2", "B"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["*"]
    assert scores.correct == 1 + 1 + 0
    assert scores.reference == 1 + 2 + 2 + 2
    assert scores.predicted == 2 + 1 + 2
    scores = named_scores["*", "<span>"]
    assert scores.correct == 1 + 1 + 2
    assert scores.reference == 1 + 2 + 2 + 1
    assert scores.predicted == 2 + 1 + 2

    named_scores = anafora.evaluate.score_data(
        reference, predicted, exclude=["X", "Y"],
        spans_type=anafora.evaluate._OverlappingSpans)
    assert set(named_scores.keys()) == {
        "*", ("*", "<span>"),
        "Z", ("Z", "<span>"), ("Z", "Source"), ("Z", "Target"), ("Z", "Prop1"), ("Z", "Prop2"),
        ("Z", "Prop1", "T"), ("Z", "Prop1", "F"), ("Z", "Prop2", "A"), ("Z", "Prop2", "B"),
        "Ref", ("Ref", "<span>"), ("Ref", "Ref"),
        }
    scores = named_scores["Z"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "<span>"]
    assert scores.correct == 2
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "Prop1"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "Prop1", "T"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 1
    scores = named_scores["Z", "Prop1", "F"]
    assert scores.correct == 0
    assert scores.reference == 0
    assert scores.predicted == 1
    scores = named_scores["Z", "Prop2"]
    assert scores.correct == 2
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "Prop2", "A"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["Z", "Prop2", "B"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["*"]
    assert scores.correct == 1
    assert scores.reference == 2 + 2
    assert scores.predicted == 2
    scores = named_scores["*", "<span>"]
    assert scores.correct == 2
    assert scores.reference == 2 + 1
    assert scores.predicted == 2

    named_scores = anafora.evaluate.score_data(
        reference, predicted, include=[("Z", "Prop1", "T")],
        spans_type=anafora.evaluate._OverlappingSpans)
    assert set(named_scores.keys()) == {("Z", "Prop1", "T")}
    scores = named_scores["Z", "Prop1", "T"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 1

    named_scores = anafora.evaluate.score_data(
        reference, predicted, include=[("Z", "Prop1", "F")],
        spans_type=anafora.evaluate._OverlappingSpans)
    assert set(named_scores.keys()) == {("Z", "Prop1", "F")}
    scores = named_scores["Z", "Prop1", "F"]
    assert scores.correct == 0
    assert scores.reference == 0
    assert scores.predicted == 1

    named_scores = anafora.evaluate.score_data(
        reference, predicted, include=["Z"], exclude=[("Z", "<span>")],
        spans_type=anafora.evaluate._OverlappingSpans)
    assert set(named_scores.keys()) == {
        "Z", ("Z", "Source"), ("Z", "Target"), ("Z", "Prop1"), ("Z", "Prop2"),
        ("Z", "Prop1", "T"), ("Z", "Prop1", "F"), ("Z", "Prop2", "A"), ("Z", "Prop2", "B"),
        }
    scores = named_scores["Z"]
    assert scores.correct == 0
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "Prop1"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "Prop1", "T"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 1
    scores = named_scores["Z", "Prop1", "F"]
    assert scores.correct == 0
    assert scores.reference == 0
    assert scores.predicted == 1
    scores = named_scores["Z", "Prop2"]
    assert scores.correct == 2
    assert scores.reference == 2
    assert scores.predicted == 2
    scores = named_scores["Z", "Prop2", "A"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["Z", "Prop2", "B"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1


def test_missing_ignored_properties():
    reference = anafora.AnaforaData(anafora.ElementTree.fromstring("""
    <data>
        <annotations>
            <entity>
                <id>4</id>
                <type>Z</type>
                <properties>
                    <A>1</A>
                    <B>2</B>
                    <C></C>
                </properties>
            </entity>
        </annotations>
    </data>
    """))
    predicted = anafora.AnaforaData(anafora.ElementTree.fromstring("""
    <data>
        <annotations>
            <entity>
                <id>4</id>
                <type>Z</type>
                <properties>
                    <B>2</B>
                    <A>1</A>
                </properties>
            </entity>
        </annotations>
    </data>
    """))
    named_scores = anafora.evaluate.score_data(reference, predicted)
    scores = named_scores["Z"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1

    # make sure no exceptions are thrown
    anafora.evaluate._print_document_scores([("temp", named_scores)])

    named_scores = anafora.evaluate.score_data(
        reference, predicted, exclude=[("Z", "C")])
    scores = named_scores["Z"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1


def test_temporal_closure_scores():

    def annotation(source, target, value):
        return (source, target), None, (None, value)

    reference = {
        annotation("A", "B", "BEFORE"),
        annotation("B", "C", "IS_INCLUDED"),
        annotation("D", "C", "INCLUDES"),
        annotation("E", "D", "CONTAINS"),
        annotation("F", "E", "AFTER"),
        annotation("G", "H", "BEGINS-ON"),
        annotation("I", "G", "BEFORE"),
        annotation("J", "K", "IBEFORE"),
        annotation("K", "L", "BEGUN_BY"),
        annotation("L", "K", "BEGINS"), # duplicate
        # inferred:
        # A before B
        # A before F
        # B before F
        # C includes B
        # C before F
        # D includes B
        # D includes C
        # D before F
        # E includes B
        # E includes C
        # E includes D
        # E before F
        # G simultaneous-start H
        # I before G
        # I before H
    }
    predicted = {
        annotation("A", "B", "BEFORE"),   # (+)
        annotation("A", "B", "BEFORE"),   # duplicate
        annotation("B", "A", "AFTER"),    # duplicate
        annotation("B", "E", "CONTAINS"), # (-)
        annotation("B", "E", "INCLUDES"), # duplicate
        annotation("B", "F", "BEFORE"),   # (+)
        annotation("F", "D", "AFTER"),    # (+)
        annotation("H", "I", "AFTER"),    # (+)
        annotation("J", "L", "IBEFORE"),  # (+)
        annotation("K", "L", "BEGUN_BY"), # (+)
        # inferred:
        # (+) A before B
        # ( ) A before E
        # ( ) A before F
        # ( ) B includes E
        # ( ) B before F
        # ( ) D before F
        # (+) E before F
        # ( ) G after I
        # ( ) J i-before L
        # (+) K begun-by L
        # (+) J i-before K
    }
    scores = anafora.evaluate.TemporalClosureScores()
    scores.add(reference, predicted)
    assert scores.precision() == 6.0 / 7.0
    assert scores.recall() ==  4.0 / 9.0


def test_temporal_closure_data():
    reference = anafora.AnaforaData(anafora.ElementTree.fromstring("""
        <data>
            <annotations>
                <entity>
                    <id>67@e@ID006_clinic_016@gold</id>
                    <span>2220,2231</span>
                    <type>TIMEX3</type>
                    <parentsType>TemporalEntities</parentsType>
                    <properties>
                        <Class>DATE</Class>
                    </properties>
                </entity>
                <entity>
                    <id>20@e@ID006_clinic_016@gold</id>
                    <span>2211,2219</span>
                    <type>EVENT</type>
                    <parentsType>TemporalEntities</parentsType>
                    <properties>
                        <DocTimeRel>BEFORE</DocTimeRel>
                        <Type>N/A</Type>
                        <Degree>N/A</Degree>
                        <Polarity>POS</Polarity>
                        <ContextualModality>ACTUAL</ContextualModality>
                        <ContextualAspect>N/A</ContextualAspect>
                        <Permanence>UNDETERMINED</Permanence>
                    </properties>
                </entity>
                <entity>
                    <id>22@e@ID006_clinic_016@gold</id>
                    <span>2273,2279</span>
                    <type>EVENT</type>
                    <parentsType>TemporalEntities</parentsType>
                    <properties>
                        <DocTimeRel>BEFORE</DocTimeRel>
                        <Type>N/A</Type>
                        <Degree>N/A</Degree>
                        <Polarity>POS</Polarity>
                        <ContextualModality>ACTUAL</ContextualModality>
                        <ContextualAspect>N/A</ContextualAspect>
                        <Permanence>UNDETERMINED</Permanence>
                    </properties>
                </entity>
                <relation>
                    <id>16@r@ID006_clinic_016@gold</id>
                    <type>TLINK</type>
                    <parentsType>TemporalRelations</parentsType>
                    <properties>
                        <Source>67@e@ID006_clinic_016@gold</Source>
                        <Type>CONTAINS</Type>
                        <Target>20@e@ID006_clinic_016@gold</Target>
                    </properties>
                </relation>
                <relation>
                    <id>36@r@ID006_clinic_016@gold</id>
                    <type>TLINK</type>
                    <parentsType>TemporalRelations</parentsType>
                    <properties>
                        <Source>20@e@ID006_clinic_016@gold</Source>
                        <Type>CONTAINS</Type>
                        <Target>22@e@ID006_clinic_016@gold</Target>
                    </properties>
                </relation>
            </annotations>
        </data>
        """))
    predicted = anafora.AnaforaData(anafora.ElementTree.fromstring("""
    <data>
        <annotations>
            <entity>
                <id>56@regex</id>
                <type>EVENT</type>
                <span>2211,2219</span>
                <properties>
                    <Polarity>POS</Polarity>
                    <Degree>N/A</Degree>
                    <Permanence>UNDETERMINED</Permanence>
                    <ContextualAspect>N/A</ContextualAspect>
                    <DocTimeRel>OVERLAP</DocTimeRel>
                    <Type>N/A</Type>
                    <ContextualModality>ACTUAL</ContextualModality>
                </properties>
            </entity>
            <entity>
                <id>57@regex</id>
                <type>TIMEX3</type>
                <span>2220,2231</span>
                <properties>
                    <Class>DATE</Class>
                </properties>
            </entity>
            <entity>
                <id>58@regex</id>
                <type>EVENT</type>
                <span>2273,2279</span>
                <properties>
                    <Polarity>POS</Polarity>
                    <Degree>N/A</Degree>
                    <Permanence>UNDETERMINED</Permanence>
                    <ContextualAspect>N/A</ContextualAspect>
                    <DocTimeRel>BEFORE/OVERLAP</DocTimeRel>
                    <Type>EVIDENTIAL</Type>
                    <ContextualModality>ACTUAL</ContextualModality>
                </properties>
            </entity>
            <relation>
                <id>57@regex@TLINK@58@regex</id>
                <type>TLINK</type>
                <properties>
                    <Source>57@regex</Source>
                    <Target>58@regex</Target>
                    <Type>CONTAINS</Type>
                    <Extra>42</Extra>
                </properties>
            </relation>
            <relation>
                <id>55@regex@TLINK@58@regex</id>
                <type>TLINK</type>
                <properties>
                    <Source>56@regex</Source>
                    <Target>58@regex</Target>
                    <Type>CONTAINS</Type>
                </properties>
            </relation>
        </annotations>
    </data>
    """))
    # reference: T(2220,2231) -> E(2211,2219) -> E(2273,2279)
    # predicted: T(2220,2231) -> E(2211,2219); E(2273,2279) -> E(2211,2219)

    named_scores = anafora.evaluate.score_data(
        reference, predicted, include={"TLINK"})
    scores = named_scores["TLINK"]
    assert scores.correct == 0
    assert scores.reference == 2
    assert scores.predicted == 2

    named_scores = anafora.evaluate.score_data(
        reference, predicted, include={("TLINK", "Type", "CONTAINS")})
    scores = named_scores["TLINK", "Type", "CONTAINS"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 2

    named_scores = anafora.evaluate.score_data(
        reference, predicted, include={("TLINK", "Type", "CONTAINS")},
        scores_type=anafora.evaluate.TemporalClosureScores)
    scores = named_scores["TLINK", "Type", "CONTAINS"]
    assert scores.precision_correct == 2
    assert scores.recall_correct == 1
    assert scores.reference == 2
    assert scores.predicted == 2

    named_scores = anafora.evaluate.score_data(
        reference, predicted, exclude={"EVENT"}, include={"TLINK"})
    scores = named_scores["TLINK"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 2

    named_scores = anafora.evaluate.score_data(
        reference, predicted, exclude={"TIMEX3"}, include={"TLINK"})
    scores = named_scores["TLINK"]
    assert scores.correct == 0
    assert scores.reference == 2
    assert scores.predicted == 2

    named_scores = anafora.evaluate.score_data(
        reference, predicted, exclude={"EVENT", "TIMEX3"}, include={"TLINK"})
    scores = named_scores["TLINK"]
    assert scores.correct == 1
    assert scores.reference == 2
    assert scores.predicted == 2

    with pytest.raises(RuntimeError) as exc_info:
        anafora.evaluate.score_data(
            reference, predicted, include={"EVENT"},
            scores_type=anafora.evaluate.TemporalClosureScores)
    assert "binary spans" in str(exc_info.value)

    with pytest.raises(RuntimeError) as exc_info:
        anafora.evaluate.score_data(
            reference, predicted, include={"TLINK"},
            scores_type=anafora.evaluate.TemporalClosureScores)
    assert "single property" in str(exc_info.value)


def test_delete_excluded():
    reference = anafora.AnaforaData(anafora.ElementTree.fromstring("""
    <data>
        <annotations>
            <entity>
                <id>1@e</id>
                <type>Z</type>
                <span>1, 3</span>
                <properties>
                    <A>2@e</A>
                </properties>
            </entity>
            <entity>
                <id>2@e</id>
                <type>Y</type>
                <span>4, 6</span>
                <properties>
                    <B>3@e</B>
                </properties>
            </entity>
            <entity>
                <id>3@e</id>
                <type>X</type>
                <span>7, 9</span>
            </entity>
            <entity>
                <id>4@e</id>
                <type>W</type>
                <span>20, 30</span>
                <properties>
                    <B>3@e</B>
                </properties>
            </entity>
        </annotations>
    </data>
    """))
    predicted = anafora.AnaforaData(anafora.ElementTree.fromstring("""
    <data>
        <annotations>
            <entity>
                <id>4@e</id>
                <type>Z</type>
                <span>1, 3</span>
                <properties>
                    <A>5@e</A>
                </properties>
            </entity>
            <entity>
                <id>5@e</id>
                <type>Y</type>
                <span>4, 6</span>
                <properties>
                    <B>6@e</B>
                </properties>
            </entity>
            <entity>
                <id>6@e</id>
                <type>X</type>
                <span>10, 15</span>
            </entity>
            <entity>
                <id>7@e</id>
                <type>W</type>
                <span>20, 30</span>
                <properties>
                    <B></B>
                </properties>
            </entity>
        </annotations>
    </data>
    """))
    named_scores = anafora.evaluate.score_data(reference, predicted)
    scores = named_scores["X"]
    assert scores.correct == 0
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["Y"]
    assert scores.correct == 0
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["Z"]
    assert scores.correct == 0
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["W"]
    assert scores.correct == 0
    assert scores.reference == 1
    assert scores.predicted == 1

    named_scores = anafora.evaluate.score_data(
        reference, predicted, exclude={"X"})
    scores = named_scores["Y"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["Z"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["W"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1

    named_scores = anafora.evaluate.score_data(
        reference, predicted, exclude={"Y"})
    scores = named_scores["X"]
    assert scores.correct == 0
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["Z"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1

    named_scores = anafora.evaluate.score_data(
        reference, predicted, exclude={"Z"})
    scores = named_scores["X"]
    assert scores.correct == 0
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["Y"]
    assert scores.correct == 0
    assert scores.reference == 1
    assert scores.predicted == 1

    named_scores = anafora.evaluate.score_data(
        reference, predicted, include={("*", "<span>")})
    scores = named_scores["X", "<span>"]
    assert scores.correct == 0
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["Y", "<span>"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["Z", "<span>"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1
    scores = named_scores["W", "<span>"]
    assert scores.correct == 1
    assert scores.reference == 1
    assert scores.predicted == 1
