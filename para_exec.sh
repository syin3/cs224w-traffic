indexof()
{
    local word
    local item
    local idx

    word=$1
    shift
    item=$(printf '%s\n' "$@" | fgrep -nx "$word")
    idx=$((${item%%:*}-1))
    return $idx
}

for var in "$@"; do
(
        indexof $var "$@"
        screen -d -m sh exec_para.sh $var $?
        )
done