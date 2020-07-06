import re

import anafora
import anafora.timeml

APW19981205_0374_tml = """<?xml version="1.0" ?>
<TimeML xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://timeml.org/timeMLdocs/TimeML_1.2.1.xsd">

<DOCID>APW19981205.0374</DOCID>

<DCT><TIMEX3 tid="t0" type="TIME" functionInDocument="CREATION_TIME" temporalFunction="false" value="1998-12-05T09:42:00">12/05/1998 09:42:00</TIMEX3></DCT>

<TITLE>Poland to participate in preparations for NATO summit</TITLE>


<EXTRAINFO>
APW19981205.0374 <DOCTYPE> NEWS STORY </DOCTYPE><DATE_TIME> 12/05/1998 09:42:00 </DATE_TIME><HEADER>w1008 &amp;Cx1f; wstm-
r i &amp;Cx13;  &amp;Cx11;  BC-Poland-Germany-NATO     12-05 0152</HEADER><SLUG> BC-Poland-Germany-NATO </SLUG><HEADLINE>Poland to participate in preparations for NATO summit</HEADLINE> &amp;UR; AP Photos WAR150-1 &amp;QL;
</EXTRAINFO>


<TEXT>

WARSAW, Poland (AP) _

Poland, Hungary and the Czech Republic can now <EVENT eid="e1" class="OCCURRENCE">participate</EVENT> in NATO strategy <EVENT eid="e24" class="OCCURRENCE">planning</EVENT>, because all NATO member states have <EVENT eid="e2" class="I_ACTION">approved</EVENT> their <EVENT eid="e3" class="STATE">joining</EVENT> the alliance, German defense minister Rudolf Scharping <EVENT eid="e4" class="REPORTING">said</EVENT> <TIMEX3 tid="t2" type="DATE" functionInDocument="NONE" temporalFunction="true" value="1998-12-05">Saturday</TIMEX3>.

The Netherlands <TIMEX3 tid="t3" type="DATE" functionInDocument="NONE" temporalFunction="true" value="1998-W49" anchorTimeID="t0">early this week</TIMEX3> <EVENT eid="e5" class="STATE">became</EVENT> the last country to <EVENT eid="e7" class="OCCURRENCE">ratify</EVENT> NATO expansion to include the three former Warsaw Pact nations.

The three are <EVENT eid="e8" class="I_STATE">expected</EVENT> to <EVENT eid="e10" class="OCCURRENCE">join</EVENT> <SIGNAL sid="s3">in</SIGNAL> <TIMEX3 tid="t4" type="DATE" functionInDocument="NONE" temporalFunction="true" value="1999-04">April</TIMEX3>, as part of <EVENT eid="e11" class="OCCURRENCE">marking</EVENT> NATO's 50th anniversary. With their membership secure, they also can <EVENT eid="e12" class="OCCURRENCE">participate</EVENT> in <EVENT eid="e13" class="I_ACTION">planning</EVENT> for the NATO <EVENT eid="e14" class="OCCURRENCE">summit</EVENT> <SIGNAL sid="s4">in</SIGNAL> <TIMEX3 tid="t5" type="DATE" functionInDocument="NONE" temporalFunction="true" value="1999-04">April</TIMEX3>.

Scharping, <EVENT eid="e15" class="ASPECTUAL">ending</EVENT> a <TIMEX3 tid="t6" type="DURATION" functionInDocument="NONE" temporalFunction="true" value="P2D">two-day</TIMEX3>  <EVENT eid="e16" class="OCCURRENCE">visit</EVENT>, <EVENT eid="e17" class="REPORTING">said</EVENT> Poland was well <EVENT eid="e18" class="I_STATE">prepared</EVENT> to <EVENT eid="e19" class="OCCURRENCE">join</EVENT> the alliance. Germany has <EVENT eid="e21" class="STATE">been</EVENT> a strong advocate of Poland's access to NATO, <EVENT eid="e22" class="REPORTING">saying</EVENT> it will <EVENT eid="e23" class="OCCURRENCE">serve</EVENT> European security and stability.

</TEXT>


<EXTRAINFO>
&amp;UR; (ms-aet)  (PROFILE (WS SL:BC-Poland-Germany-NATO; CT:i; (REG:EURO;) (REG:BRIT;) (REG:SCAN;) (REG:ENGL;) (LANG:ENGLISH;)) )
<TRAILER> AP-NY-12-05-98 0942EST </TRAILER>
</EXTRAINFO>


<MAKEINSTANCE eiid="ei1" eventID="e1" tense="PRESENT" aspect="NONE" pos="UNKNOWN" modality="can" polarity="POS"/>
<MAKEINSTANCE eiid="ei2" eventID="e2" tense="PAST" aspect="PERFECTIVE" pos="UNKNOWN" polarity="POS"/>
<MAKEINSTANCE eiid="ei3" eventID="e3" tense="NONE" aspect="NONE" pos="NOUN" polarity="POS"/>
<MAKEINSTANCE eiid="ei4" eventID="e4" tense="PAST" aspect="NONE" pos="UNKNOWN" polarity="POS"/>
<MAKEINSTANCE eiid="ei5" eventID="e5" tense="PAST" aspect="NONE" pos="UNKNOWN" polarity="POS"/>
<MAKEINSTANCE eiid="ei7" eventID="e7" tense="INFINITIVE" aspect="NONE" pos="VERB" polarity="POS"/>
<MAKEINSTANCE eiid="ei9" eventID="e8" tense="PRESENT" aspect="NONE" pos="UNKNOWN" polarity="POS"/>
<MAKEINSTANCE eiid="ei12" eventID="e10" tense="INFINITIVE" aspect="NONE" pos="VERB" polarity="POS"/>
<MAKEINSTANCE eiid="ei13" eventID="e11" tense="NONE" aspect="NONE" pos="NOUN" polarity="POS"/>
<MAKEINSTANCE eiid="ei14" eventID="e12" tense="PRESENT" aspect="NONE" pos="UNKNOWN" modality="can" polarity="POS"/>
<MAKEINSTANCE eiid="ei15" eventID="e13" tense="NONE" aspect="NONE" pos="NOUN" polarity="POS"/>
<MAKEINSTANCE eiid="ei16" eventID="e14" tense="NONE" aspect="NONE" pos="NOUN" polarity="POS"/>
<MAKEINSTANCE eiid="ei17" eventID="e15" tense="PRESPART" aspect="NONE" pos="VERB" polarity="POS"/>
<MAKEINSTANCE eiid="ei18" eventID="e16" tense="NONE" aspect="NONE" pos="NOUN" polarity="POS"/>
<MAKEINSTANCE eiid="ei19" eventID="e17" tense="PAST" aspect="NONE" pos="UNKNOWN" polarity="POS"/>
<MAKEINSTANCE eiid="ei20" eventID="e18" tense="PAST" aspect="NONE" pos="UNKNOWN" polarity="POS"/>
<MAKEINSTANCE eiid="ei21" eventID="e19" tense="INFINITIVE" aspect="NONE" pos="VERB" polarity="POS"/>
<MAKEINSTANCE eiid="ei23" eventID="e21" tense="PRESENT" aspect="PERFECTIVE" pos="UNKNOWN" polarity="POS"/>
<MAKEINSTANCE eiid="ei24" eventID="e22" tense="PRESPART" aspect="NONE" pos="VERB" polarity="POS"/>
<MAKEINSTANCE eiid="ei25" eventID="e23" tense="FUTURE" aspect="NONE" pos="UNKNOWN" polarity="POS"/>
<MAKEINSTANCE eiid="ei26" eventID="e24" tense="NONE" aspect="NONE" pos="NOUN" polarity="POS"/>
<TLINK lid="l2" timeID="t0" relatedToEventInstance="ei1" relType="INCLUDES" origin="USER"/>
<TLINK lid="l3" eventInstanceID="ei2" relatedToEventInstance="ei1" relType="BEFORE" origin="USER"/>
<TLINK lid="l5" timeID="t0" relatedToEventInstance="ei2" relType="AFTER" origin="USER"/>
<TLINK lid="l6" timeID="t2" relatedToEventInstance="ei4" relType="INCLUDES" origin="USER"/>
<TLINK lid="l7" timeID="t2" relatedToEventInstance="ei4" relType="INCLUDES" origin="USER"/>
<TLINK lid="l8" eventInstanceID="ei4" relatedToEventInstance="ei2" relType="AFTER" origin="USER"/>
<TLINK lid="l9" timeID="t3" relatedToEventInstance="ei5" relType="INCLUDES" origin="USER"/>
<TLINK lid="l11" eventInstanceID="ei5" relatedToEventInstance="ei7" relType="SIMULTANEOUS" origin="USER"/>
<TLINK lid="l12" timeID="t0" relatedToEventInstance="ei9" relType="INCLUDES" origin="USER"/>
<TLINK lid="l14" timeID="t4" signalID="s3" relatedToEventInstance="ei12" relType="INCLUDES" origin="USER"/>
<TLINK lid="l15" timeID="t4" relatedToEventInstance="ei13" relType="INCLUDES" origin="USER"/>
<TLINK lid="l16" timeID="t0" relatedToEventInstance="ei13" relType="BEFORE" origin="USER"/>
<TLINK lid="l17" timeID="t0" relatedToEventInstance="ei14" relType="INCLUDES" origin="USER"/>
<TLINK lid="l20" timeID="t5" signalID="s4" relatedToEventInstance="ei16" relType="INCLUDES" origin="USER"/>
<TLINK lid="l21" timeID="t6" relatedToEventInstance="ei18" relType="INCLUDES" origin="USER"/>
<TLINK lid="l23" timeID="t2" relatedToEventInstance="ei19" relType="INCLUDES" origin="USER"/>
<TLINK lid="l24" timeID="t6" relatedToEventInstance="ei19" relType="DURING" origin="USER"/>
<TLINK lid="l26" eventInstanceID="ei3" relatedToTime="t4" relType="INCLUDES" origin="USER"/>
<TLINK lid="l27" timeID="t0" relatedToEventInstance="ei26" relType="BEFORE" origin="USER"/>
<TLINK lid="l28" timeID="t0" relatedToEventInstance="ei15" relType="BEFORE" origin="USER"/>
<TLINK lid="l29" timeID="t2" relatedToEventInstance="ei17" relType="AFTER" origin="USER"/>
<TLINK lid="l32" timeID="t4" relatedToEventInstance="ei21" relType="INCLUDES" origin="USER"/>
<TLINK lid="l33" timeID="t0" relatedToEventInstance="ei20" relType="AFTER" origin="USER"/>
<TLINK lid="l35" timeID="t0" relatedToEventInstance="ei23" relType="AFTER" origin="USER"/>
<TLINK lid="l36" timeID="t0" relatedToEventInstance="ei24" relType="AFTER" origin="USER"/>
<TLINK lid="l38" timeID="t0" relatedToEventInstance="ei25" relType="BEFORE" origin="USER"/>
<SLINK lid="l1" eventInstanceID="ei2" subordinatedEventInstance="ei3" relType="MODAL"/>
<SLINK lid="l10" eventInstanceID="ei1" subordinatedEventInstance="ei26" relType="MODAL"/>
<SLINK lid="l13" eventInstanceID="ei9" subordinatedEventInstance="ei12" relType="MODAL"/>
<SLINK lid="l18" eventInstanceID="ei14" subordinatedEventInstance="ei15" relType="MODAL"/>
<SLINK lid="l19" eventInstanceID="ei15" subordinatedEventInstance="ei16" relType="MODAL"/>
<SLINK lid="l30" eventInstanceID="ei19" subordinatedEventInstance="ei20" relType="EVIDENTIAL"/>
<SLINK lid="l31" eventInstanceID="ei20" subordinatedEventInstance="ei21" relType="MODAL"/>
<SLINK lid="l37" eventInstanceID="ei24" subordinatedEventInstance="ei25" relType="EVIDENTIAL"/>
<ALINK lid="l22" eventInstanceID="ei18" relatedToEventInstance="ei17" relType="CULMINATES"/>
</TimeML>
"""


def test_to_text(tmpdir):

    # remove elements
    text = re.sub(r"<[^>]*>", "", APW19981205_0374_tml)
    # replace XML entities
    text = text.replace("&amp;", "&")
    # remove spaces before and after root element
    text = text[1:-1]

    # test that method produces the same thing
    path = tmpdir.join("APW19981205_0374.tml")
    path.write(APW19981205_0374_tml)
    assert anafora.timeml.to_text(str(path)) == text


def test_to_document_creation_time(tmpdir):
    path = tmpdir.join("APW19981205_0374.tml")
    path.write(APW19981205_0374_tml)
    assert anafora.timeml.to_document_creation_time(str(path)) == "1998-12-05T09:42:00"



def test_to_anafora_data(tmpdir):
    path = tmpdir.join("APW19981205_0374.tml")
    path.write(APW19981205_0374_tml)

    text = anafora.timeml.to_text(str(path))
    data = anafora.timeml.to_anafora_data(str(path))

    # check counts of various annotations
    assert len(list(data.annotations.select_type("EVENT"))) == 21
    assert len(list(data.annotations.select_type("TIMEX3"))) == 6
    assert len(list(data.annotations.select_type("SIGNAL"))) == 2
    assert len(list(data.annotations.select_type("TLINK"))) == 26
    assert len(list(data.annotations.select_type("SLINK"))) == 8
    assert len(list(data.annotations.select_type("ALINK"))) == 1

    # <TIMEX3 tid="t0" type="TIME" functionInDocument="CREATION_TIME" temporalFunction="false"
    #         value="1998-12-05T09:42:00">12/05/1998 09:42:00</TIMEX3>
    annotation = data.annotations.select_id("1@e@APW19981205_0374@gold")
    for start, end in annotation.spans:
        assert text[start:end] == "12/05/1998 09:42:00"
    pattern = "^<entity><id>.*?</id><type>TIMEX3</type><span>20,39</span><properties>.*?</properties></entity>$"
    assert re.match(pattern, str(annotation))
    assert dict(annotation.properties.items()) == {
        "type": "TIME",
        "functionInDocument": "CREATION_TIME",
        "temporalFunction": "false",
        "value": "1998-12-05T09:42:00",
    }

    # <TIMEX3 tid="t3" type="DATE" functionInDocument="NONE" temporalFunction="true" value="1998-W49"
    #         anchorTimeID="t0">early this week</TIMEX3>
    annotation = data.annotations.select_id("8@e@APW19981205_0374@gold")
    for start, end in annotation.spans:
        assert text[start:end] == "early this week"
    pattern = "^<entity><id>.*?</id><type>TIMEX3</type><span>.*?</span><properties>.*?</properties></entity>$"
    assert re.match(pattern, str(annotation))
    assert dict(annotation.properties.items()) == {
        "type": "DATE",
        "functionInDocument": "NONE",
        "temporalFunction": "true",
        "value": "1998-W49",
        "anchorTimeID": data.annotations.select_id("1@e@APW19981205_0374@gold"),
    }

    # <EVENT eid="e8" class="I_STATE">expected</EVENT>
    annotation = data.annotations.select_id("11@e@APW19981205_0374@gold")
    for start, end in annotation.spans:
        assert text[start:end] == "expected"
    pattern = "^<entity><id>.*?</id><type>EVENT</type><span>.*?</span><properties>.*?</properties></entity>$"
    assert re.match(pattern, str(annotation))
    assert dict(annotation.properties.items()) == {"class": "I_STATE"}

    # <SIGNAL sid="s3">in</SIGNAL>
    annotation = data.annotations.select_id("13@e@APW19981205_0374@gold")
    for start, end in annotation.spans:
        assert text[start:end] == "in"
    pattern = "^<entity><id>.*?</id><type>SIGNAL</type><span>.*?</span></entity>$"
    assert re.match(pattern, str(annotation))
    assert not annotation.properties.items()

    # <MAKEINSTANCE eiid="ei23" eventID="e21" tense="PRESENT" aspect="PERFECTIVE" pos="UNKNOWN" polarity="POS"/>
    annotation = data.annotations.select_id("47@r@APW19981205_0374@gold")
    pattern = "^<relation><id>.*?</id><type>MAKEINSTANCE</type><properties>.*?</properties></relation>"
    assert re.match(pattern, str(annotation))
    assert dict(annotation.properties.items()) == {
        "eventID": data.annotations.select_id("27@e@APW19981205_0374@gold"),
        "tense": "PRESENT",
        "aspect": "PERFECTIVE",
        "pos": "UNKNOWN",
        "polarity": "POS",
    }
    assert dict(annotation.properties["eventID"].properties.items()) == {"class": "STATE"}

    # <TLINK lid="l9" timeID="t3" relatedToEventInstance="ei5" relType="INCLUDES" origin="USER"/>
    annotation = data.annotations.select_id("57@r@APW19981205_0374@gold")
    pattern = "^<relation><id>.*?</id><type>TLINK</type><properties>.*?</properties></relation>"
    assert re.match(pattern, str(annotation))
    assert dict(annotation.properties.items()) == {
        "timeID": data.annotations.select_id("8@e@APW19981205_0374@gold"),
        "relatedToEventInstance": data.annotations.select_id("34@r@APW19981205_0374@gold"),
        "relType": "INCLUDES",
        "origin": "USER",
    }
    assert dict(annotation.properties["timeID"].properties.items()) == {
        "type": "DATE",
        "functionInDocument": "NONE",
        "temporalFunction": "true",
        "value": "1998-W49",
        "anchorTimeID": data.annotations.select_id("1@e@APW19981205_0374@gold"),
    }
    assert dict(annotation.properties["relatedToEventInstance"].properties.items()) == {
        "eventID": data.annotations.select_id("9@e@APW19981205_0374@gold"),
        "tense": "PAST",
        "aspect": "NONE",
        "pos": "UNKNOWN",
        "polarity": "POS",
    }

    # <SLINK lid="l37" eventInstanceID="ei24" subordinatedEventInstance="ei25" relType="EVIDENTIAL"/>
    annotation = data.annotations.select_id("84@r@APW19981205_0374@gold")
    pattern = "^<relation><id>.*?</id><type>SLINK</type><properties>.*?</properties></relation>"
    assert re.match(pattern, str(annotation))
    assert dict(annotation.properties.items()) == {
        "eventInstanceID": data.annotations.select_id("48@r@APW19981205_0374@gold"),
        "subordinatedEventInstance": data.annotations.select_id("49@r@APW19981205_0374@gold"),
        "relType": "EVIDENTIAL",
    }

    # <ALINK lid="l22" eventInstanceID="ei18" relatedToEventInstance="ei17" relType="CULMINATES"/>
    annotation = data.annotations.select_id("85@r@APW19981205_0374@gold")
    pattern = "^<relation><id>.*?</id><type>ALINK</type><properties>.*?</properties></relation>"
    assert re.match(pattern, str(annotation))
    assert dict(annotation.properties.items()) == {
        "eventInstanceID": data.annotations.select_id("43@r@APW19981205_0374@gold"),
        "relatedToEventInstance": data.annotations.select_id("42@r@APW19981205_0374@gold"),
        "relType": "CULMINATES",
    }
