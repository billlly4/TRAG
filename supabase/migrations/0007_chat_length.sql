-- Per-chat length limit
--
-- Chats live in Postgres, not in the Storage bucket, so the 100 MB file quota
-- from 0006 does not constrain them at all. They are a real consumer: every
-- search stores the full text of the retrieved passages in a tool_result block
-- so the conversation can be replayed to a stateless API, which makes a single
-- message row up to ~60 KB. Measured on this database, 26 messages already
-- outweighed 40 chunks.
--
-- Counted in message ROWS, not conversational turns. One exchange that uses the
-- search tool writes four rows: the user's question, the assistant's tool call,
-- the tool results, and the final answer. So the default 200 is roughly 50
-- exchanges, or about 1 MB of message JSON per chat.

insert into public.quotas (key, value) values
  ('max_messages_per_thread', 200)
on conflict (key) do nothing;

-- SECURITY DEFINER for the same reason as the other quota triggers: the count
-- must be right regardless of whose policies are in play. It filters by
-- NEW.thread_id, so it can only ever count one conversation.
create or replace function public.enforce_thread_length()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  limit_value bigint;
  current_count bigint;
begin
  select value into limit_value
    from public.quotas where key = 'max_messages_per_thread';
  if limit_value is null then
    return new;
  end if;

  select count(*) into current_count
    from public.messages
   where thread_id = new.thread_id;

  if current_count >= limit_value then
    raise exception 'thread_length_exceeded: % of % messages already in this chat',
      current_count, limit_value
      using errcode = 'check_violation';
  end if;

  return new;
end;
$$;

drop trigger if exists messages_enforce_length on public.messages;
create trigger messages_enforce_length
  before insert on public.messages
  for each row execute function public.enforce_thread_length();

-- ---------------------------------------------------------------------------
-- Verification (run manually in the SQL editor)
--
--   -- 1. The limit is registered alongside the others
--   select * from public.quotas order by key;
--
--   -- 2. Trigger attached
--   select tgname from pg_trigger where tgname = 'messages_enforce_length';
--
--   -- 3. Longest chats, and how much JSON they hold
--   select thread_id, count(*) as messages,
--          pg_size_pretty(sum(pg_column_size(content))::bigint) as content
--     from public.messages group by thread_id order by messages desc limit 5;
--
--   -- To change the limit, update the row -- no function rewrite needed:
--   --   update public.quotas set value = 400 where key = 'max_messages_per_thread';
-- ---------------------------------------------------------------------------
