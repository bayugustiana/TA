<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateTransaksiTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('transaksi', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->unsignedBigInteger('id_peternakan');
            $table->unsignedBigInteger('id_obat');
            $table->integer('total_harga');
            $table->integer('packaging');
            $table->date('tanggal_transaksi');
            $table->enum('status', ['diproses','lunas']);
            $table->foreign('id_peternakan')->references('id')->on('peternakan')->onUpdate('cascade')->onDelete('cascade');
            $table->foreign('id_obat')->references('id')->on('obat')->onUpdate('cascade')->onDelete('cascade');

        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('transaksi');
    }
}
