import gradio as gr
from sidekick import Sidekick
from user_auth import UserAuth

# Initialize auth system
auth = UserAuth()

async def setup(username):
    """Setup Sidekick for specific user"""
    sidekick = Sidekick(user_id=username)
    await sidekick.setup()
    return sidekick

async def process_message(sidekick, message, success_criteria, history):
    """Process user message"""
    results = await sidekick.run_superstep(message, success_criteria, history)
    # Clear input fields after processing
    return results, sidekick, "", ""

async def reset(username):
    """Reset conversation for user"""
    new_sidekick = Sidekick(user_id=username)
    await new_sidekick.setup()
    return "", "", None, new_sidekick

def free_resources(sidekick):
    """Cleanup resources"""
    print("Cleaning up")
    try:
        if sidekick:
            sidekick.cleanup()
    except Exception as e:
        print(f"Exception during cleanup: {e}")

def handle_login(username, password):
    """Handle user login"""
    success, message = auth.login(username, password)
    if success:
        return (
            gr.update(visible=False),  # Hide login box
            gr.update(visible=True),   # Show main app
            username,                   # Set current user
            f"Logged in as: {username}",  # User display
            message  # Login status
        )
    else:
        return (
            gr.update(visible=True),   # Keep login box visible
            gr.update(visible=False),  # Keep main app hidden
            None,                       # No user
            "",                         # No user display
            message  # Error message
        )

def handle_register(username, password):
    """Handle user registration"""
    success, message = auth.register(username, password)
    return message

def handle_logout():
    """Handle user logout"""
    return (
        gr.update(visible=True),   # Show login box
        gr.update(visible=False),  # Hide main app
        None,                       # Clear current user
        "",                         # Clear user display
        "",                         # Clear message
        "",                         # Clear success_criteria
        None,                       # Clear chatbot
        "Logged out successfully"  # Login status
    )

# Build Gradio UI
with gr.Blocks(title="Sidekick Multi-User") as ui:
    gr.Markdown("# 🤖 Sidekick Personal Co-Worker")

    # State management
    current_user = gr.State(None)
    sidekick = gr.State(delete_callback=free_resources)

    # Login/Register Box
    with gr.Group(visible=True) as login_box:
        gr.Markdown("## 🔐 Login or Register")

        with gr.Tab("Login"):
            login_username = gr.Textbox(
                label="Username",
                placeholder="Enter your username"
            )
            login_password = gr.Textbox(
                label="Password",
                type="password",
                placeholder="Enter your password"
            )
            login_btn = gr.Button("Login", variant="primary")
            login_status = gr.Textbox(label="Status", interactive=False)

        with gr.Tab("Register"):
            reg_username = gr.Textbox(
                label="Username",
                placeholder="Choose a username (min 3 characters)"
            )
            reg_password = gr.Textbox(
                label="Password",
                type="password",
                placeholder="Choose a password (min 6 characters)"
            )
            reg_btn = gr.Button("Register", variant="primary")
            reg_status = gr.Textbox(label="Status", interactive=False)

    # Main App (hidden until login)
    with gr.Group(visible=False) as main_app:
        gr.Markdown("## 💬 Chat with Sidekick")

        with gr.Row():
            user_display = gr.Textbox(
                label="Current User",
                interactive=False,
                scale=3
            )
            logout_btn = gr.Button("Logout", variant="stop", scale=1)

        with gr.Row():
            chatbot = gr.Chatbot(label="Conversation", height=400)

        with gr.Group():
            with gr.Row():
                message = gr.Textbox(
                    show_label=False,
                    placeholder="Your request to the Sidekick...",
                    scale=4
                )
            with gr.Row():
                success_criteria = gr.Textbox(
                    show_label=False,
                    placeholder="What are your success criteria?",
                    scale=4
                )

        with gr.Row():
            reset_button = gr.Button("Reset Conversation", variant="secondary")
            go_button = gr.Button("Go!", variant="primary")

    # Event handlers - Login/Register
    login_btn.click(
        handle_login,
        [login_username, login_password],
        [login_box, main_app, current_user, user_display, login_status]
    ).then(
        setup,
        [current_user],
        [sidekick]
    )

    reg_btn.click(
        handle_register,
        [reg_username, reg_password],
        [reg_status]
    )

    # Event handlers - Logout
    logout_btn.click(
        handle_logout,
        None,
        [login_box, main_app, current_user, user_display, message, success_criteria, chatbot, login_status]
    )

    # Event handlers - Main App
    message.submit(
        process_message,
        [sidekick, message, success_criteria, chatbot],
        [chatbot, sidekick, message, success_criteria]
    )

    success_criteria.submit(
        process_message,
        [sidekick, message, success_criteria, chatbot],
        [chatbot, sidekick, message, success_criteria]
    )

    go_button.click(
        process_message,
        [sidekick, message, success_criteria, chatbot],
        [chatbot, sidekick, message, success_criteria]
    )

    reset_button.click(
        reset,
        [current_user],
        [message, success_criteria, chatbot, sidekick]
    )

# Launch
ui.launch(inbrowser=True, theme=gr.themes.Default(primary_hue="emerald"))