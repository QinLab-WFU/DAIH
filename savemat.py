import os
import scipy.io as scio





def Save_mat(epoch, output_dim, datasets, query_img, retrieval_img, query_labels, retrieval_labels, save_dir='..', mode_name="DSLAH",map=0):
    '''
    save_dir: 保存文件的目录路径
    output_dim: 输出维度
    datasets: 数据集名称
    query_labels: 查询图像的标签信息（numpy数组）
    retrieval_labels: 检索图像的标签信息（numpy数组）
    query_img: 查询图像的数据（numpy数组）
    retrieval_img: 检索图像的数据（numpy数组）
    mode_name: 模型的名称
    '''
    save_dir = os.path.join(save_dir , f"Hash_code_and_label_{mode_name}_{datasets}")
    os.makedirs(save_dir,exist_ok=True)

    query_img = query_img.cpu().detach().numpy()
    retrieval_img = retrieval_img.cpu().detach().numpy()

    query_label = query_labels.cpu().detach().numpy()
    retrieval_label = retrieval_labels.cpu().detach().numpy()

    result_dict = {
        'q_img' : query_img ,
        'r_img' : retrieval_img ,
        'q_l' : query_label ,
        'r_l' : retrieval_label
    }
    filename = os.path.join(save_dir, f"{output_dim}-{epoch}-{datasets}-{mode_name}-{map}.mat")
    scio.savemat(filename, result_dict)

def save_mat(epoch, datasets, query_img, query_txt, retrieval_img, retrieval_txt,
             query_labels, retrieval_labels, save_dir='..', mode_name="i2t", map=0):

        save_dir = os.path.join(save_dir, f"Hash_code_and_label_{mode_name}_{datasets}")
        os.makedirs(save_dir, exist_ok=True)

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = query_labels.cpu().detach().numpy()
        retrieval_labels = retrieval_labels.cpu().detach().numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        filename = os.path.join(save_dir, f"{128}-{epoch}-{datasets}-{mode_name}-{map}.mat")
        scio.savemat(filename, result_dict)