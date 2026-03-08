# ==========================================
# 2. 图形化用户界面 (GUI) 控制台 - 购物车累加版
# ==========================================
def create_gui():
    root = tk.Tk()
    root.title("T-SFE Molecular Docking Station")
    root.geometry("550x350") # 稍微拉长一点窗口，放得下新按钮
    
    root.eval('tk::PlaceWindow . center')

    receptor_var = tk.StringVar()
    ligands_var = tk.StringVar(value="No ligands added.")
    ligands_paths_list = []

    def select_receptor():
        filepath = filedialog.askopenfilename(
            title="🎯 Step 1: Select RNA Target", 
            filetypes=[("PDB Files", "*.pdb")]
        )
        if filepath:
            receptor_var.set(filepath)

    def add_ligands():
        # 核心修改：改为累加模式
        filepaths = filedialog.askopenfilenames(
            title="💊 Step 2: Add Ligand", 
            filetypes=[("SDF Files", "*.sdf")]
        )
        if filepaths:
            for f in filepaths:
                if f not in ligands_paths_list: # 防止重复添加同一个药
                    ligands_paths_list.append(f)
            
            names = [os.path.basename(f) for f in ligands_paths_list]
            ligands_var.set(f"Ready ({len(names)}): " + ", ".join(names))

    def clear_ligands():
        # 新增：清空按钮逻辑
        ligands_paths_list.clear()
        ligands_var.set("No ligands added.")

    def start_engine():
        receptor = receptor_var.get()
        if not receptor or not ligands_paths_list:
            messagebox.showwarning("⚠️ Warning", "Please select ONE target and AT LEAST ONE ligand!")
            return
        
        messagebox.showinfo("Engine Started", "Docking initiated! Check PyCharm console.")
        root.destroy() 
        
        run_docking_engine(receptor, ligands_paths_list)

    # UI 布局排版
    tk.Label(root, text="Step 1: Load Target", font=("Arial", 12, "bold")).pack(pady=(10, 5))
    tk.Button(root, text="📂 Select RNA (.pdb)", command=select_receptor, width=25, bg="#e0e0e0").pack()
    tk.Label(root, textvariable=receptor_var, fg="blue").pack()

    tk.Label(root, text="Step 2: Load Ligands (Add one by one)", font=("Arial", 12, "bold")).pack(pady=(10, 5))
    
    # 把“添加”和“清空”按钮放在同一行
    btn_frame = tk.Frame(root)
    btn_frame.pack()
    tk.Button(btn_frame, text="➕ Add Ligand (.sdf)", command=add_ligands, width=15, bg="#e0e0e0").pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="🗑️ Clear List", command=clear_ligands, width=15, bg="#ffb3b3").pack(side=tk.LEFT, padx=5)
    
    # 显示已添加的配体列表
    tk.Label(root, textvariable=ligands_var, fg="green", wraplength=500).pack(pady=5)

    tk.Button(root, text="🚀 Start Docking!", command=start_engine, width=25, height=2, bg="#ff4d4d", fg="white", font=("Arial", 12, "bold")).pack(pady=15)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
