sts() {
    local user="nbyrnes"  # Replace "your_username" with the actual username

    local server_name="$1"
    local destination="$2"
    shift 2
    local source_files="$@"
    
    # Set the server address based on the server name
    local server_address=""
    case "$server_name" in
        "jupiter")
            server_address="$user@otg-jupiter.dynip.ntu.edu.sg"
            ;;
        "mercury")
            server_address="$user@otg-mercury.dynip.ntu.edu.sg"
            ;;
        "mars")
            server_address="$user@otg-mars.dynip.ntu.edu.sg"
            ;;
        *)
            echo "Unknown server name: $server_name"
            return 1
            ;;
    esac

    local get_files=false
    while getopts "g" opt; do
        case "$opt" in
            g)
                get_files=true
                ;;
            *)
                ;;
        esac
    done

    # Perform the appropriate rsync operation based on the -g option
    if [ "$get_files" = true ]; then
        rsync -avz "$server_address:$destination/" "${source_files[@]}"
        echo "File(s) received successfully from $server_name."
    else
        rsync -avz "${source_files[@]}" "$server_address:$destination/"
        echo "File(s) sent successfully to $server_name."
    fi
}