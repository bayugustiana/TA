<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateObatTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('obat', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->unsignedBigInteger('id_toko_obat');
            $table->string('tipe_obat');
            $table->string('nama_obat');
            $table->float('harga_obat');
            $table->text('deskripsi');
            $table->integer('stok');
            $table->string('avatar');
            $table->foreign('id_toko_obat')->references('id')->on('toko_obat')->onUpdate('cascade')->onDelete('cascade');
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('obat');
    }
}
