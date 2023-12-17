import os

max_result_files = 5
result_files_buffer = []

def save_results_to_file(results, filename):
    with open(filename, 'w') as file:
        for result in results:
            file.write(result + '\n')

def delete_old_result_files():
    global result_files_buffer
    while len(result_files_buffer) > max_result_files:
        file_to_delete = result_files_buffer.pop(0)
        os.remove(file_to_delete)

captions = []

with torch.inference_mode():
    for idx, batch in enumerate(tqdm(dataloader)):
        try:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["pixel_values"]).detach().cpu().numpy()
            captions.extend(mapping_output_tensor_to_tags(outputs, id2label=id2label, threshold=0.5))

            # Save results after each successful iteration
            result_filename = f"results_batch_{idx}.txt"
            save_results_to_file(captions, result_filename)
            result_files_buffer.append(result_filename)
            
            # Delete old result files to maintain the buffer limit
            delete_old_result_files()
        except Exception as e:
            # Handle the exception (e.g., print an error message)
            print(f"Error processing batch {idx}: {str(e)}")
            # Optionally, save the results even if an error occurs
            result_filename = f"results_batch_{idx}_error.txt"
            save_results_to_file(captions, result_filename)
            result_files_buffer.append(result_filename)
            
            # Delete old result files to maintain the buffer limit
            delete_old_result_files()
