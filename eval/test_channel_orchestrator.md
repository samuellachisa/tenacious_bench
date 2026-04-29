# Channel Orchestrator Test Specification

## Overview

Tests for `agent/core/channel_orchestrator.py` - the central state machine that manages channel transitions and eligibility.

## Test Cases

### 1. Channel Eligibility Tests

#### Test 1.1: Email eligibility at all stages
```python
def test_email_always_eligible():
    orchestrator = ChannelOrchestrator(mock_obs)
    
    stages = [
        ProspectStage.NEW,
        ProspectStage.OUTBOUND_SENT,
        ProspectStage.EMAIL_OPENED,
        ProspectStage.REPLIED,
        ProspectStage.QUALIFIED,
        ProspectStage.SCHEDULED,
        ProspectStage.CALL_BOOKED,
    ]
    
    for stage in stages:
        result = orchestrator.check_channel_eligibility(Channel.EMAIL, stage)
        assert result.eligible == True
        assert "Email available" in result.reason
```

#### Test 1.2: SMS only for warm leads
```python
def test_sms_warm_lead_only():
    orchestrator = ChannelOrchestrator(mock_obs)
    
    # Cold stages - SMS not eligible
    cold_stages = [ProspectStage.NEW, ProspectStage.OUTBOUND_SENT]
    for stage in cold_stages:
        result = orchestrator.check_channel_eligibility(Channel.SMS, stage)
        assert result.eligible == False
        assert "Cold lead" in result.reason
    
    # Warm stages - SMS eligible
    warm_stages = [
        ProspectStage.EMAIL_OPENED,
        ProspectStage.REPLIED,
        ProspectStage.QUALIFIED,
        ProspectStage.SCHEDULED,
        ProspectStage.CALL_BOOKED,
    ]
    for stage in warm_stages:
        result = orchestrator.check_channel_eligibility(Channel.SMS, stage)
        assert result.eligible == True
        assert "Warm lead" in result.reason
```

#### Test 1.3: Cal.com only after qualification
```python
def test_calcom_qualified_only():
    orchestrator = ChannelOrchestrator(mock_obs)
    
    # Pre-qualification - Cal.com not eligible
    early_stages = [
        ProspectStage.NEW,
        ProspectStage.OUTBOUND_SENT,
        ProspectStage.EMAIL_OPENED,
        ProspectStage.REPLIED,
    ]
    for stage in early_stages:
        result = orchestrator.check_channel_eligibility(Channel.CALCOM, stage)
        assert result.eligible == False
        assert "requires qualification" in result.reason
    
    # Post-qualification - Cal.com eligible
    qualified_stages = [
        ProspectStage.QUALIFIED,
        ProspectStage.SCHEDULED,
        ProspectStage.CALL_BOOKED,
    ]
    for stage in qualified_stages:
        result = orchestrator.check_channel_eligibility(Channel.CALCOM, stage)
        assert result.eligible == True
        assert "Qualified stage" in result.reason
```

#### Test 1.4: CRM always eligible
```python
def test_crm_always_eligible():
    orchestrator = ChannelOrchestrator(mock_obs)
    
    all_stages = list(ProspectStage)
    
    for stage in all_stages:
        result = orchestrator.check_channel_eligibility(Channel.CRM, stage)
        assert result.eligible == True
        assert "CRM updates available" in result.reason
```

#### Test 1.5: Disqualified blocks email
```python
def test_disqualified_blocks_email():
    orchestrator = ChannelOrchestrator(mock_obs)
    
    result = orchestrator.check_channel_eligibility(
        Channel.EMAIL,
        ProspectStage.DISQUALIFIED
    )
    
    assert result.eligible == False
    assert "disqualified" in result.reason.lower()
```

### 2. State Transition Tests

#### Test 2.1: Valid forward transitions
```python
def test_valid_forward_transitions():
    orchestrator = ChannelOrchestrator(mock_obs)
    
    valid_paths = [
        (ProspectStage.NEW, ProspectStage.OUTBOUND_SENT),
        (ProspectStage.OUTBOUND_SENT, ProspectStage.EMAIL_OPENED),
        (ProspectStage.EMAIL_OPENED, ProspectStage.REPLIED),
        (ProspectStage.REPLIED, ProspectStage.QUALIFIED),
        (ProspectStage.QUALIFIED, ProspectStage.SCHEDULED),
        (ProspectStage.SCHEDULED, ProspectStage.CALL_BOOKED),
    ]
    
    for from_stage, to_stage in valid_paths:
        result = orchestrator.transition_stage(from_stage, to_stage, "TestCo")
        assert result.success == True
        assert result.from_stage == from_stage
        assert result.to_stage == to_stage
        assert len(result.allowed_channels) > 0
```

#### Test 2.2: Invalid backward transitions
```python
def test_invalid_backward_transitions():
    orchestrator = ChannelOrchestrator(mock_obs)
    
    # Can't go backward in pipeline
    result = orchestrator.transition_stage(
        ProspectStage.QUALIFIED,
        ProspectStage.REPLIED,
        "TestCo"
    )
    
    assert result.success == False
    assert "Invalid transition" in result.reason
    assert result.allowed_channels == []
```

#### Test 2.3: Skip-stage transitions
```python
def test_skip_stage_transitions():
    orchestrator = ChannelOrchestrator(mock_obs)
    
    # Can skip from OUTBOUND_SENT directly to REPLIED (prospect replies immediately)
    result = orchestrator.transition_stage(
        ProspectStage.OUTBOUND_SENT,
        ProspectStage.REPLIED,
        "TestCo"
    )
    
    assert result.success == True
    assert Channel.SMS in result.allowed_channels  # Now warm lead
```

#### Test 2.4: Disqualification from any stage
```python
def test_disqualification_from_any_stage():
    orchestrator = ChannelOrchestrator(mock_obs)
    
    stages = [
        ProspectStage.NEW,
        ProspectStage.OUTBOUND_SENT,
        ProspectStage.EMAIL_OPENED,
        ProspectStage.REPLIED,
        ProspectStage.QUALIFIED,
        ProspectStage.SCHEDULED,
        ProspectStage.CALL_BOOKED,
    ]
    
    for stage in stages:
        result = orchestrator.transition_stage(
            stage,
            ProspectStage.DISQUALIFIED,
            "TestCo"
        )
        assert result.success == True
```

#### Test 2.5: Terminal state enforcement
```python
def test_terminal_states():
    orchestrator = ChannelOrchestrator(mock_obs)
    
    # Can't transition from DISQUALIFIED
    result = orchestrator.transition_stage(
        ProspectStage.DISQUALIFIED,
        ProspectStage.QUALIFIED,
        "TestCo"
    )
    
    assert result.success == False
    
    # Can't transition from CALL_BOOKED (except to disqualified)
    result = orchestrator.transition_stage(
        ProspectStage.CALL_BOOKED,
        ProspectStage.SCHEDULED,
        "TestCo"
    )
    
    assert result.success == False
```

### 3. Next Action Recommendation Tests

#### Test 3.1: Email opened triggers followup
```python
def test_email_opened_action():
    orchestrator = ChannelOrchestrator(mock_obs)
    
    result = orchestrator.get_next_action(
        current_stage=ProspectStage.OUTBOUND_SENT,
        engagement_signal="email_opened",
        company="TestCo"
    )
    
    assert result["action"] == "send_followup_email"
    assert result["next_stage"] == "email_opened"
    assert "email" in result["channels"]
    assert "sms" in result["channels"]  # Now warm lead
```

#### Test 3.2: Reply triggers qualification
```python
def test_reply_action():
    orchestrator = ChannelOrchestrator(mock_obs)
    
    result = orchestrator.get_next_action(
        current_stage=ProspectStage.EMAIL_OPENED,
        engagement_signal="email_replied",
        company="TestCo"
    )
    
    assert result["action"] == "qualify_prospect"
    assert result["next_stage"] == "replied"
    assert "sms" in result["channels"]
```

#### Test 3.3: Qualification triggers Cal.com
```python
def test_qualification_action():
    orchestrator = ChannelOrchestrator(mock_obs)
    
    result = orchestrator.get_next_action(
        current_stage=ProspectStage.REPLIED,
        engagement_signal="qualified",
        company="TestCo"
    )
    
    assert result["action"] == "send_calcom_link"
    assert result["next_stage"] == "qualified"
    assert "calcom" in result["channels"]
    assert "sms" in result["channels"]
```

#### Test 3.4: Scheduled triggers SMS reminder
```python
def test_scheduled_action():
    orchestrator = ChannelOrchestrator(mock_obs)
    
    result = orchestrator.get_next_action(
        current_stage=ProspectStage.QUALIFIED,
        engagement_signal="scheduled",
        company="TestCo"
    )
    
    assert result["action"] == "send_sms_reminder"
    assert result["next_stage"] == "scheduled"
    assert "sms" in result["channels"]
```

#### Test 3.5: No signal maintains stage
```python
def test_no_signal_maintains_stage():
    orchestrator = ChannelOrchestrator(mock_obs)
    
    result = orchestrator.get_next_action(
        current_stage=ProspectStage.OUTBOUND_SENT,
        engagement_signal=None,
        company="TestCo"
    )
    
    assert result["action"] == "maintain"
    assert result["current_stage"] == result["next_stage"]
    assert "email" in result["channels"]
    assert "sms" not in result["channels"]  # Still cold
```

### 4. Integration Tests

#### Test 4.1: Full pipeline flow
```python
def test_full_pipeline_flow():
    orchestrator = ChannelOrchestrator(mock_obs)
    company = "TestCo"
    
    # Stage 1: Send initial email
    action = orchestrator.get_next_action(ProspectStage.NEW, None, company)
    assert action["action"] == "maintain"
    
    transition = orchestrator.transition_stage(ProspectStage.NEW, ProspectStage.OUTBOUND_SENT, company)
    assert transition.success == True
    assert Channel.SMS not in transition.allowed_channels
    
    # Stage 2: Email opened
    action = orchestrator.get_next_action(ProspectStage.OUTBOUND_SENT, "email_opened", company)
    assert action["action"] == "send_followup_email"
    assert "sms" in action["channels"]  # Now warm
    
    # Stage 3: Prospect replies
    action = orchestrator.get_next_action(ProspectStage.EMAIL_OPENED, "email_replied", company)
    assert action["action"] == "qualify_prospect"
    
    # Stage 4: Qualified
    action = orchestrator.get_next_action(ProspectStage.REPLIED, "qualified", company)
    assert action["action"] == "send_calcom_link"
    assert "calcom" in action["channels"]
    
    # Stage 5: Scheduled
    action = orchestrator.get_next_action(ProspectStage.QUALIFIED, "scheduled", company)
    assert action["action"] == "send_sms_reminder"
```

#### Test 4.2: Fast-track reply (skip email_opened)
```python
def test_fast_track_reply():
    orchestrator = ChannelOrchestrator(mock_obs)
    
    # Prospect replies immediately without opening email first
    transition = orchestrator.transition_stage(
        ProspectStage.OUTBOUND_SENT,
        ProspectStage.REPLIED,
        "TestCo"
    )
    
    assert transition.success == True
    assert Channel.SMS in transition.allowed_channels
```

## Expected Outcomes

All tests should pass with:
- Channel eligibility correctly enforced at each stage
- Invalid transitions rejected with clear reasons
- Next action recommendations aligned with stage
- Observability traces logged for all transitions
- Fail-closed behavior (reject when uncertain)

## Monitoring

Track in Langfuse:
- `channel_orchestrator_transition` - All valid transitions
- `channel_orchestrator_invalid_transition` - Rejected transitions
- `channel_orchestrator_next_action` - Action recommendations

## Integration Points

The orchestrator should be used by:
1. `agent/main.py` - Email reply handler
2. `agent/domain/use_cases/qualify_prospect.py` - After qualification
3. `agent/integrations/sms_client.py` - Before sending SMS
4. `agent/integrations/calcom_client.py` - Before booking calls
5. `agent/adapters/gateways/hubspot_crm_adapter.py` - Before stage updates
