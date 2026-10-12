use neura_template::{Message, Template, Value};

const QWEN: &str = include_str!("fixtures/qwen35.jinja");

fn refuses<T>(action: impl FnOnce() -> T) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn qwen() -> Template {
    Template::parse(QWEN)
}

fn message(role: &str, content: &str) -> Message {
    Message::new(role, content)
}

fn generation() -> [(&'static str, Value); 2] {
    [
        ("add_generation_prompt", Value::Flag(true)),
        ("enable_thinking", Value::Flag(false)),
    ]
}

fn tool_call() -> Value {
    Value::List(vec![Value::Map(vec![(
        "function".to_owned(),
        Value::Map(vec![
            ("name".to_owned(), Value::Text("get_weather".to_owned())),
            (
                "arguments".to_owned(),
                Value::Map(vec![("city".to_owned(), Value::Text("Paris".to_owned()))]),
            ),
        ]),
    )])])
}

fn weather_tool() -> Value {
    let city = Value::Map(vec![("type".to_owned(), Value::Text("string".to_owned()))]);
    let parameters = Value::Map(vec![
        ("type".to_owned(), Value::Text("object".to_owned())),
        (
            "properties".to_owned(),
            Value::Map(vec![("city".to_owned(), city)]),
        ),
        (
            "required".to_owned(),
            Value::List(vec![Value::Text("city".to_owned())]),
        ),
    ]);
    let function = Value::Map(vec![
        ("name".to_owned(), Value::Text("get_weather".to_owned())),
        (
            "description".to_owned(),
            Value::Text("Get the weather".to_owned()),
        ),
        ("parameters".to_owned(), parameters),
    ]);
    Value::List(vec![Value::Map(vec![
        ("type".to_owned(), Value::Text("function".to_owned())),
        ("function".to_owned(), function),
    ])])
}

#[test]
fn a_question_draws_the_prompt_the_model_was_trained_on() {
    let held = qwen().render(
        &[message("user", "What is the capital of France?")],
        &generation(),
    );
    assert_eq!(held, include_str!("fixtures/qwen35-single.txt"));
}

#[test]
fn every_turn_of_a_conversation_draws_its_own_markers() {
    let held = qwen().render(
        &[
            message("user", "Hello"),
            message("assistant", "Hi there"),
            message("user", "How are you?"),
        ],
        &[
            ("add_generation_prompt", Value::Flag(true)),
            ("enable_thinking", Value::Flag(true)),
        ],
    );
    assert_eq!(held, include_str!("fixtures/qwen35-turn.txt"));
}

#[test]
fn a_system_message_opens_the_prompt() {
    let held = qwen().render(
        &[
            message("system", "You are a helpful assistant."),
            message("user", "Hello"),
        ],
        &generation(),
    );
    assert_eq!(held, include_str!("fixtures/qwen35-system.txt"));
}

#[test]
fn the_reasoning_of_an_earlier_answer_stays_out_of_the_prompt() {
    let held = qwen().render(
        &[
            message("user", "What is 2+2?"),
            message("assistant", "<think>\nLet me add.\n</think>\n\nFour."),
            message("user", "Thanks"),
        ],
        &generation(),
    );
    assert_eq!(held, include_str!("fixtures/qwen35-reasoning.txt"));
}

#[test]
fn a_call_and_its_answer_walk_as_turns_of_their_own() {
    let held = qwen().render(
        &[
            message("user", "Weather in Paris?"),
            message("assistant", "").with("tool_calls", tool_call()),
            message("tool", "18C and sunny"),
        ],
        &generation(),
    );
    assert_eq!(held, include_str!("fixtures/qwen35-tool.txt"));
}

#[test]
fn the_tools_on_offer_stand_ahead_of_the_question() {
    let held = qwen().render(
        &[message("user", "Weather in Paris?")],
        &[
            ("add_generation_prompt", Value::Flag(true)),
            ("enable_thinking", Value::Flag(false)),
            ("tools", weather_tool()),
        ],
    );
    assert_eq!(held, include_str!("fixtures/qwen35-tools.txt"));
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
        "neither"
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
    assert_eq!(
        template.render(
            &[],
            &[(
                "held",
                Value::List(vec![
                    Value::Text("a".to_owned()),
                    Value::Text("b".to_owned()),
                ]),
            )],
        ),
        "abab!",
    );
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
    assert_eq!(template.render(&[], &[("held", Value::Flag(true))]), "yes",);
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
        refuses(|| qwen().render(&[], &generation())),
        "the template raises on a conversation that holds no message",
    );
}
