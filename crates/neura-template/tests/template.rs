use neura_template::{Message, Template, Value};

fn refuses<T>(action: impl FnOnce() -> T) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn transcript() -> Template {
    Template::parse(include_str!("fixtures/transcript.jinja"))
}

fn quiet() -> [(&'static str, Value); 1] {
    [("answer_thinking", Value::Flag(false))]
}

fn loud() -> [(&'static str, Value); 1] {
    [("answer_thinking", Value::Flag(true))]
}

fn item(text: &str) -> Value {
    Value::Map(vec![("text".to_owned(), Value::Text(text.to_owned()))])
}

#[test]
fn a_question_renders_the_turn_it_asks() {
    let held = transcript().render(&[Message::new("user", "what is the plan?")], &quiet());
    assert_eq!(held, "tone: few\nuser 1: what is the plan?\naside (none) ",);
}

#[test]
fn the_rules_and_the_earlier_turns_stand_on_their_own_lines() {
    let held = transcript().render(
        &[
            Message::new("system", "the rules"),
            Message::new("user", "who goes there?"),
        ],
        &quiet(),
    );
    assert_eq!(
        held,
        "tone: few\n<rules>the rules</rules>\nuser 2: who goes there?\naside (none) ",
    );
    let held = transcript().render(
        &[
            Message::new("user", "hello"),
            Message::new("assistant", "hi"),
            Message::new("user", "how are you?"),
        ],
        &loud(),
    );
    assert_eq!(
        held,
        "tone: many\nuser 1: hello\nhi\nuser 3: how are you?\naside ",
    );
}

#[test]
fn the_thought_of_an_answer_stays_out_of_its_turn() {
    let held = transcript().render(
        &[
            Message::new("user", "what is two plus two?"),
            Message::new("assistant", "<thinking>\nadd them\n<<\nfour."),
            Message::new("user", "thanks"),
        ],
        &quiet(),
    );
    assert_eq!(
        held,
        "tone: many\nuser 1: what is two plus two?\nfour.\nuser 3: thanks\naside (none) ",
    );
}

#[test]
fn consecutive_notes_open_once_and_close_once() {
    let held = transcript().render(
        &[
            Message::new("user", "list them"),
            Message::new("assistant", "sure"),
            Message::new("note", "first"),
            Message::new("note", "second"),
            Message::new("user", "and then?"),
        ],
        &quiet(),
    );
    assert_eq!(
        held,
        "tone: many\nuser 1: list them\nsure\nnotes: firstsecond;\nuser 5: and then?\naside (none) ",
    );
}

#[test]
fn a_turn_reads_the_items_a_body_holds() {
    let held = transcript().render(
        &[Message::new(
            "user",
            Value::List(vec![
                item("one"),
                Value::Map(vec![("count".to_owned(), Value::Integer(3))]),
                item("two"),
            ]),
        )],
        &quiet(),
    );
    assert_eq!(
        held,
        "tone: few\nuser 1: one{\"count\": 3}two\naside (none) "
    );
}

#[test]
fn the_thought_of_the_last_turn_stands_apart() {
    let held = transcript().render(
        &[
            Message::new("user", "hello"),
            Message::new("assistant", "hello").with("thought", "quietly"),
        ],
        &loud(),
    );
    assert_eq!(
        held,
        "tone: few\nuser 1: hello\naside[quietly] hello\naside ",
    );
}

#[test]
fn a_turn_of_no_content_leaves_its_line_empty() {
    let held = transcript().render(
        &[
            Message::new("assistant", Value::Empty),
            Message::new("user", "?"),
        ],
        &quiet(),
    );
    assert_eq!(held, "tone: few\n\nuser 2: ?\naside (none) ");
}

#[test]
fn a_transcript_the_template_cannot_read_is_refused() {
    assert!(refuses(|| transcript().render(&[], &quiet())));
    assert!(refuses(
        || transcript().render(&[Message::new("assistant", "hi")], &quiet())
    ));
    assert!(refuses(|| transcript().render(
        &[Message::new("user", "hi"), Message::new("tool", "anywhere"),],
        &quiet(),
    )));
    assert!(refuses(|| transcript().render(
        &[Message::new(
            "user",
            Value::List(vec![Value::Map(vec![(
                "picture".to_owned(),
                Value::Text("a picture".to_owned()),
            )])]),
        )],
        &quiet(),
    )));
    assert!(refuses(|| transcript().render(
        &[Message::new(
            "user",
            Value::Map(vec![("count".to_owned(), Value::Integer(3))]),
        )],
        &quiet(),
    )));
}

#[test]
fn a_branch_prints_the_arm_its_condition_picks() {
    let template =
        Template::parse("{% if held %}{{ held }}{% elif other %}neither{% else %}none{% endif %}");
    assert_eq!(
        template.render(&[], &[("held", Value::Text("here".to_owned()))]),
        "here",
    );
    assert_eq!(
        template.render(&[], &[("other", Value::Flag(true))]),
        "neither",
    );
    assert_eq!(template.render(&[], &[]), "none");
}

#[test]
fn a_loop_walks_what_it_names_and_counts_itself() {
    let template = Template::parse("{% for x in held %}{{ loop.index }}{{ x }}{% endfor %}");
    let held = Value::List(vec![
        Value::Text("a".to_owned()),
        Value::Text("b".to_owned()),
    ]);
    assert_eq!(template.render(&[], &[("held", held)]), "1a2b");
    let template = Template::parse(
        "{% for x in held %}{{ loop.previtem }}{{ x }}{{ loop.nextitem }}{% if loop.last %}!{% endif %}{% endfor %}",
    );
    let held = Value::List(vec![
        Value::Text("a".to_owned()),
        Value::Text("b".to_owned()),
    ]);
    assert_eq!(template.render(&[], &[("held", held)]), "abab!");
}

#[test]
fn a_namespace_carries_what_a_loop_counts() {
    let template = Template::parse(
        "{% set ns = namespace(total=0) %}{% for x in held %}{% set ns.total = ns.total + 1 %}{% endfor %}total {{ ns.total }}",
    );
    let held = Value::List(vec![
        Value::Integer(1),
        Value::Integer(2),
        Value::Integer(3),
    ]);
    assert_eq!(template.render(&[], &[("held", held)]), "total 3");
}

#[test]
fn a_macro_returns_the_text_its_body_draws() {
    let template = Template::parse(
        "{% macro render(content, tag='p') %}<{{ tag }}>{{ content }}</{{ tag }}>{% endmacro %}{{ render('hi') }}|{{ render('ho', tag='b') }}",
    );
    assert_eq!(template.render(&[], &[]), "<p>hi</p>|<b>ho</b>");
}

#[test]
fn a_value_reads_through_the_filters_it_names() {
    let template = Template::parse("{{ held | trim }}|{{ held | length }}|{{ none | string }}");
    assert_eq!(
        template.render(&[], &[("held", Value::Text(" hi ".to_owned()))]),
        "hi|4|None",
    );
}

#[test]
fn a_value_reads_through_the_tests_it_names() {
    let template = Template::parse(
        "{{ held is defined }}{{ other is undefined }}{{ held is string }}{{ none is none }}{{ num is mapping }}{{ seq is not string }}",
    );
    let held = [
        ("held", Value::Text("x".to_owned())),
        ("num", Value::Integer(2)),
        ("seq", Value::List(vec![])),
    ];
    assert_eq!(template.render(&[], &held), "TrueTrueTrueTrueFalseTrue");
}

#[test]
fn a_list_walks_backwards_through_a_slice() {
    let template = Template::parse("{% for x in held[::-1] %}{{ x }}{% endfor %}");
    let held = Value::List(vec![
        Value::Text("a".to_owned()),
        Value::Text("b".to_owned()),
        Value::Text("c".to_owned()),
    ]);
    assert_eq!(template.render(&[], &[("held", held)]), "cba");
}

#[test]
fn the_condition_of_a_question_picks_between_two_texts() {
    let template = Template::parse("{{ 'yes' if held else 'no' }}");
    assert_eq!(template.render(&[], &[("held", Value::Flag(true))]), "yes");
    assert_eq!(template.render(&[], &[]), "no");
}

#[test]
fn a_mapping_answers_whether_it_holds_a_key() {
    let template = Template::parse("{{ 'city' in held }}{{ 'town' in held }}");
    let held = Value::Map(vec![("city".to_owned(), Value::Text("Paris".to_owned()))]);
    assert_eq!(template.render(&[], &[("held", held)]), "TrueFalse");
}

#[test]
fn the_whitespace_a_dash_trims_stays_out_of_the_text() {
    let template = Template::parse("a\n  {%- if held -%}\n  b\n  {%- endif -%}\n  c");
    assert_eq!(template.render(&[], &[("held", Value::Flag(true))]), "abc");
    let template = Template::parse("{# a comment #}kept");
    assert_eq!(template.render(&[], &[]), "kept");
}

#[test]
fn a_value_renders_as_json_the_way_a_template_writes_it() {
    let template = Template::parse("{{ held | tojson }}");
    let held = Value::Map(vec![
        ("b".to_owned(), Value::Integer(1)),
        (
            "a".to_owned(),
            Value::List(vec![Value::Integer(1), Value::Text("\u{e9}<".to_owned())]),
        ),
    ]);
    assert_eq!(
        template.render(&[], &[("held", held)]),
        r#"{"a": [1, "\u00e9\u003c"], "b": 1}"#,
    );
}

#[test]
fn a_tag_this_renderer_reads_no_part_of_is_refused() {
    assert!(refuses(|| Template::parse("{% include 'other.jinja' %}")));
    assert!(refuses(|| Template::parse("{% for %}{% endfor %}")));
    assert!(refuses(|| Template::parse("{% if held %}")));
    assert!(refuses(|| Template::parse("{{ held ")));
    assert!(refuses(
        || Template::parse("{{ held | untold }}").render(&[], &[])
    ));
    assert!(refuses(
        || Template::parse("{{ held is untold }}").render(&[], &[])
    ));
    assert!(refuses(|| Template::parse("{{ call() }}").render(&[], &[])));
    assert!(refuses(|| Template::parse("{{ held.untold() }}")
        .render(&[], &[("held", Value::Text("x".to_owned()))],)));
    assert!(
        refuses(|| Template::parse("{{ raise_exception('no') }}").render(&[], &[])),
        "a template raises through the same call every renderer reads",
    );
}
